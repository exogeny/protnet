from typing import Callable, List, Any, Tuple, Dict, Optional
import torch
from torch import nn, Tensor
from einops import rearrange
from xformers.ops import fmha, scaled_index_add, index_select_cat

from prot.layers import Attention
from prot.layers import MemEffAttention
from prot.layers import DropPath
from prot.layers import LayerScale
from prot.layers import Mlp
from prot.layers import Mamba2

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py



class BlockChunk(nn.ModuleList):
  def forward(self, x, **kwargs):
    for b in self:
      x = b(x, **kwargs)
    return x

  @staticmethod
  def create(block_chunks, depth, blocks):
    if block_chunks > 0:
      chunked_blocks = []
      chunksize = depth // block_chunks
      for i in range(0, depth, chunksize):
        # this is to keep the block index consistent if we chunk the block list
        chunked_blocks.append([nn.Identity()] * i + blocks[i : i + chunksize])
      return nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
    return nn.ModuleList(blocks)


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
  # 1) extract subset using permutation
  b, n, d = x.shape
  sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
  brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
  x_subset = x[brange]

  # 2) apply residual_func to get residual
  residual = residual_func(x_subset)
  x_flat = x.flatten(1)
  residual = residual.flatten(1)
  residual_scale_factor = b / sample_subset_size

  # 3) add the residual
  x_plus_residual = torch.index_add(
      x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
  return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
  b, n, d = x.shape
  sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
  brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
  residual_scale_factor = b / sample_subset_size
  return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
  if scaling_vector is None:
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
  else:
    x_plus_residual = scaled_index_add(
        x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
    )
  return x_plus_residual


class Block(nn.Module):
  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float = 4.0,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      ffn_bias: bool = True,
      drop: float = 0.0,
      attn_drop: float = 0.0,
      init_values=None,
      drop_path: float = 0.0,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      attn_class: Callable[..., nn.Module] = Attention,
      ffn_layer: Callable[..., nn.Module] = Mlp,
  ) -> None:
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = attn_class(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=drop,
    )
    self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
    self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = ffn_layer(
        in_features=dim,
        hidden_features=mlp_hidden_dim,
        act_layer=act_layer,
        drop=drop,
        bias=ffn_bias,
    )
    self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
    self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    self.sample_drop_ratio = drop_path

  def forward(self, x: Tensor) -> Tensor:
    def attn_residual_func(x: Tensor) -> Tensor:
      return self.ls1(self.attn(self.norm1(x)))

    def ffn_residual_func(x: Tensor) -> Tensor:
      return self.ls2(self.mlp(self.norm2(x)))

    if self.training and self.sample_drop_ratio > 0.1:
      # the overhead is compensated only for a drop path rate larger than 0.1
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=attn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=ffn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
    elif self.training and self.sample_drop_ratio > 0.0:
      x = x + self.drop_path1(attn_residual_func(x))
      x = x + self.drop_path2(ffn_residual_func(x))
    else:
      x = x + attn_residual_func(x)
      x = x + ffn_residual_func(x)
    return x


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None, have_seq_idx=False):
  """
  this will perform the index select, cat the tensors, and provide the attn_bias from cache
  """
  batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
  all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
  if all_shapes not in attn_bias_cache.keys():
    seqlens, seq_pos = [], []
    seq_idx = None
    if have_seq_idx:
      for s in all_shapes:
        seq_pos.extend([[i + len(seq_pos)] * s[1] * 2 for i in range(s[0])])
      seq_idx = torch.tensor([i for j in seq_pos for i in j]).reshape((1, -1))
      seq_idx = seq_idx.int().to(x_list[0].device)
    for b, x in zip(batch_sizes, x_list):
      for _ in range(b):
        seqlens.append(x.shape[1])
    attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
    attn_bias._batch_sizes = batch_sizes
    attn_bias_cache[all_shapes] = (attn_bias, seq_idx)

  if branges is not None:
    cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
  else:
    tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
    cat_tensors = torch.cat(tensors_bs1, dim=1)

  attn_bias = attn_bias_cache[all_shapes] if have_seq_idx else attn_bias_cache[all_shapes][0]
  return attn_bias, cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
    has_seq_idx=False,
) -> Tensor:
  # 1) generate random set of indices for dropping samples in the batch
  branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
  branges = [s[0] for s in branges_scales]
  residual_scale_factors = [s[1] for s in branges_scales]

  # 2) get attention bias and index+concat the tensors

  # 3) apply residual_func to get residual, and split the result
  if has_seq_idx:
    (attn_bias, seq_idx), x_cat = get_attn_bias_and_cat(x_list, branges, have_seq_idx=has_seq_idx)
    residual_list = attn_bias.split(residual_func(x_cat, seq_idx=seq_idx))  # type: ignore
  else:
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

  outputs = []
  for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
    outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
  return outputs


class NestedTensorBlock(Block):
  def forward_nested(self, x_list: List[Tensor], **kwargs) -> List[Tensor]:
    """
    x_list contains a list of tensors to nest together and run
    """
    if self.training and self.sample_drop_ratio > 0.0:
      def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
        if isinstance(self.attn, MemEffAttention):
          return self.attn(self.norm1(x), attn_bias=attn_bias)
        return self.attn(self.norm1(x))

      def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
        return self.mlp(self.norm2(x))

      x_list = drop_add_residual_stochastic_depth_list(
          x_list,
          residual_func=attn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
      )
      x_list = drop_add_residual_stochastic_depth_list(
          x_list,
          residual_func=ffn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
      )
      return x_list
    else:
      def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
          if isinstance(self.attn, MemEffAttention):
              return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
          return self.ls1(self.attn(self.norm1(x)))

      def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
          return self.ls2(self.mlp(self.norm2(x)))

      attn_bias, x = get_attn_bias_and_cat(x_list)
      x = x + attn_residual_func(x, attn_bias=attn_bias)
      x = x + ffn_residual_func(x)
      x = attn_bias.split(x)
      return x

  def forward(self, x_or_x_list, **kwargs):
    if isinstance(x_or_x_list, Tensor):
      output = super().forward(x_or_x_list)
      return output
    elif isinstance(x_or_x_list, list):
      return self.forward_nested(x_or_x_list, **kwargs)
    else:
      raise AssertionError


class MambaBlock(nn.Module):
  def __init__(
      self,
      dim,
      mixer_cls,
      mlp_cls,
      norm_cls=nn.LayerNorm,
      drop_path: float = 0.,
      init_values: float = None,
      num_tokens: int = 0,
  ):
    super().__init__()
    self.num_tokens = num_tokens
    self.norm1 = norm_cls(dim)
    self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
    self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    self.mixer = mixer_cls(dim)
    if mlp_cls is not nn.Identity:
      self.norm2 = norm_cls(dim)
      self.mlp = mlp_cls(dim)
      self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
      self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    else:
      self.norm2 = None
      self.mlp = None
      self.ls2 = None
      self.drop_path2 = None
    if self.num_tokens > 1:
      self.cls_linear = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
    self.sample_drop_ratio = drop_path
    self.support_varlen_seq = isinstance(self.mixer, Mamba2)

  def interleave_cls_patch_flexible(self, x: Tensor):
    seg_lengths = None
    if self.num_tokens > 1:
      N = x.shape[1] - self.num_tokens
      M = self.num_tokens
      base = N // (M - 1)
      remainder = N % (M - 1)

      # Example: N=7, M=4 → base=2, remainder=1 → lens = [3,2,2]
      seg_lengths = [base + 1 if i < remainder else base for i in range(M - 1)]
      segments = []
      start = M
      for i in range(M):
        segments.append(x[:, i:i+1, :])  # insert cls_i
        if i < M - 1:
            length = seg_lengths[i]
            end = start + length
            segments.append(x[:, start:end, :])
            start = end
      x = torch.cat(segments, dim=1).contiguous()
    return x, seg_lengths

  def de_interleave_cls_patch_flexible(self, x: Tensor, seg_lengths: List[int]):
    if seg_lengths is not None:
      M = len(seg_lengths) + 1
      idx = 0
      cls_tokens = []
      patch_tokens = []

      for i in range(M):
        cls_tokens.append(x[:, idx:idx+1, :])
        idx += 1
        if i < M - 1:
          patch_len = seg_lengths[i]
          patch_tokens.append(x[:, idx:idx+patch_len, :])
          idx += patch_len
      x = torch.cat(
          [
              torch.cat(cls_tokens, dim=1),
              torch.cat(patch_tokens, dim=1),
          ], dim=1).contiguous()
    return x

  def split_and_cat_mixer(self, x: Tensor, seq_idx: Optional[Tensor] = None) -> Tensor:
    x, seg_lengths = self.interleave_cls_patch_flexible(x)
    x = torch.cat([x, x.flip([1])], dim=1)
    x = self.mixer(x, seq_idx=seq_idx)
    x = torch.tensor_split(x, 2, dim=1)
    x = (1 + torch.tanh(x[0])) * x[1].flip([1])
    x = self.de_interleave_cls_patch_flexible(x, seg_lengths)
    if self.num_tokens > 1:
      cls_tokens, patch_tokens = x.tensor_split([self.num_tokens], dim=1)
      cls_tokens = rearrange(
          self.cls_linear(
              rearrange(cls_tokens, 'b n d -> b d n')),
          'b d n -> b n d'
      )
      x = torch.cat([cls_tokens, patch_tokens], dim=1)
    return x

  def forward_nested(self, x_list):
    if self.training and self.sample_drop_ratio > 0.0 and self.support_varlen_seq:
      def attn_residual_func(x: Tensor, seq_idx=None) -> Tensor:
        return self.split_and_cat_mixer(self.norm1(x), seq_idx=seq_idx)

      def ffn_residual_func(x: Tensor, **_) -> Tensor:
        return self.mlp(self.norm2(x))

      x_list = drop_add_residual_stochastic_depth_list(
          x_list,
          residual_func=attn_residual_func,
          has_seq_idx=True,
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
      )
      if self.mlp is not None:
        x_list = drop_add_residual_stochastic_depth_list(
            x_list,
            residual_func=ffn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
            scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
        )
      return x_list
    else:
      def attn_residual_func(x: Tensor, seq_idx=None) -> Tensor:
        return self.ls1(self.split_and_cat_mixer(self.norm1(x), seq_idx=seq_idx))

      def ffn_residual_func(x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))

      if self.training and self.sample_drop_ratio > 0.0:
        # training and not support_varlen_seq
        x_list = [xi + self.drop_path1(attn_residual_func(xi)) for xi in x_list]
        if self.mlp is not None:
          x_list = [xi + self.drop_path2(ffn_residual_func(xi)) for xi in x_list]
      elif self.support_varlen_seq:
        (attn_bias, seq_idx), x = get_attn_bias_and_cat(x_list, have_seq_idx=True)
        x = x + attn_residual_func(x, seq_idx=seq_idx)
        if self.mlp is not None:
          x = x + ffn_residual_func(x)
        x_list = attn_bias.split(x)
      else:
        x_list = [xi + attn_residual_func(xi) for xi in x_list]
        if self.mlp is not None:
          x_list = [xi + ffn_residual_func(xi) for xi in x_list]
      return x_list

  def forward(self, x: Tensor) -> Tensor:
    if isinstance(x, list):
      return self.forward_nested(x)

    def attn_residual_func(x: Tensor) -> Tensor:
      return self.ls1(self.split_and_cat_mixer(self.norm1(x)))

    def ffn_residual_func(x: Tensor) -> Tensor:
      return self.ls2(self.mlp(self.norm2(x)))

    if self.training and self.sample_drop_ratio > 0.1 and self.support_varlen_seq:
      # the overhead is compensated only for a drop path rate larger than 0.1
      x = drop_add_residual_stochastic_depth(
          x,
          residual_func=attn_residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
      if self.mlp is not None:
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=ffn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
        )
    elif self.training and self.sample_drop_ratio > 0.0:
      x = x + self.drop_path1(attn_residual_func(x))
      if self.mlp is not None:
        x = x + self.drop_path2(ffn_residual_func(x))
    else:
      x = x + attn_residual_func(x)
      if self.mlp is not None:
        x = x + ffn_residual_func(x)
    return x
