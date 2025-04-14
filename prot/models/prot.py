from typing import Sequence, Tuple, Union, Callable, Optional
import math
import logging
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from prot.layers import Mlp, GatedMlp
from prot.layers import PatchEmbed
from prot.layers import SwiGLUFFNFused
from prot.layers import Mamba, Mamba2
from prot.layers import MemEffAttention
from prot.layers import MambaBlock
from prot.layers import BlockChunk
from prot.layers import NestedTensorBlock as AttentionBlock
from prot.ops._triton.layer_norm import RMSNorm

logger = logging.getLogger('prot')


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = '',
    depth_first: bool = True,
    include_root: bool = False
) -> nn.Module:
  if not depth_first and include_root:
    fn(module=module, name=name)
  for child_name, child_module in module.named_children():
    child_name = '.'.join((name, child_name)) if name else child_name
    named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
  if depth_first and include_root:
    fn(module=module, name=name)
  return module


def create_block(
    block_name: str,
    dim: int,
    num_heads: int,
    ffn_layer: str,
    act_layer: Callable,
    norm_layer: Callable,
    mlp_ratio: float,
    qkv_bias: bool,
    proj_bias: bool,
    ffn_bias: bool,
    drop_path: Optional[float] = None,
    init_values: Optional[float] = None,
    layer_idx: Optional[int] = None,
):
  # create mlp layer
  if mlp_ratio > 0:
    ffn_layer = (ffn_layer or 'mlp').lower()
    if ffn_layer == 'mlp':
      ffn_layer = Mlp
    elif ffn_layer == 'gatedmlp':
      ffn_layer = partial(
          GatedMlp,
          hidden_features=int(dim * mlp_ratio),
          out_features=dim,
      )
    elif ffn_layer == 'swiglufused' or ffn_layer == 'swiglu':
      ffn_layer = SwiGLUFFNFused
    else:
      ffn_layer = nn.Identity
  else:
    ffn_layer = nn.Identity

  if block_name in ['mamba1', 'mamba2']:
    if block_name == 'mamba1':
      mixer_cls = partial(
          Mamba,
          d_state=16,
          layer_idx=layer_idx
      )
    else:
      mixer_cls = partial(
          Mamba2,
          d_state=16,
          headdim=dim // num_heads,
          use_mem_eff_path=False,
          activation='silu',
          conv_bias=False,
          layer_idx=layer_idx,
          learnable_init_states=True,
          chunk_size=16,
          ngroups=1,
      )

    block = MambaBlock(
        dim=dim,
        mixer_cls=mixer_cls,
        mlp_cls=ffn_layer,
        norm_cls=norm_layer,
        drop_path=drop_path,
        init_values=init_values,
    )
    block.layer_idx = layer_idx
    return block
  elif block_name == 'attention':
    return AttentionBlock(
        dim=dim,
        num_heads=num_heads,
        attn_class=MemEffAttention,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        ffn_bias=ffn_bias,
        drop_path=drop_path,
        norm_layer=norm_layer,
        act_layer=act_layer,
        ffn_layer=ffn_layer,
        init_values=init_values,
    )
  raise ValueError(f'Attention class {block_name} not supported')


def create_blocks(
      block_chunks: int,
      depth: int,
      *arg,
      drop_paths: Sequence[float],
      layer_idx_offset: int = 0,
      **kwargs
):
  return BlockChunk.create(
      block_chunks, depth,
      [
          create_block(
              *arg,
              drop_path=drop_paths[i],
              layer_idx=i + layer_idx_offset,
              **kwargs
          ) for i in range(depth)
      ]
  )


class ProtNet(nn.Module):
  def __init__(
      self,
      img_size=224,
      patch_size=16,
      embed_dim=768,
      depth=12,
      num_heads=12,
      contour_ratio=0.25,
      decoder_embed_dim=512,
      decoder_depth=8,
      decoder_num_heads=16,
      mlp_ratio=4.0,
      bias=True,
      qkv_bias=True,
      ffn_bias=True,
      proj_bias=True,
      drop_path_rate=0.0,
      drop_path_uniform=False,
      init_values=None,  # for layerscale: None or 0 => no layerscale
      block_name='mamba2',
      embed_layer=PatchEmbed,
      act_layer=nn.GELU,
      ffn_layer='GatedMlp',
      norm_layer=partial(RMSNorm, eps=1e-5),
      block_chunks=0,
      num_tokens: int = 1,
      num_register_tokens: int = 0,
      interpolate_antialias: bool = False,
      interpolate_offset: float = 0.1,
      reconstruction_mode: bool = True,
      generation_mode: bool = True,
  ):
    """
    Args:
        img_size (int, tuple): input image size
        patch_size (int, tuple): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        proj_bias (bool): enable bias for proj in attn if True
        ffn_bias (bool): enable bias for ffn if True
        drop_path_rate (float): stochastic depth rate
        drop_path_uniform (bool): apply uniform drop rate across blocks
        weight_init (str): weight init scheme
        init_values (float): layer-scale init values
        embed_layer (nn.Module): patch embedding layer
        act_layer (nn.Module): MLP activation layer
        block_fn (nn.Module): transformer block class
        ffn_layer (str): 'mlp', 'swiglu', 'swiglufused' or 'identity'
        block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        num_register_tokens: (int) number of extra cls tokens (so-called 'registers')
        interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
        interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
    """
    super().__init__()
    self.block_name = block_name

    self.reconstruction_mode = reconstruction_mode
    self.generation_mode = generation_mode

    self.contour_embed_dim = int(embed_dim * contour_ratio)
    self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    self.num_tokens = num_tokens
    self.n_blocks = depth
    self.n_decoder_blocks = decoder_depth
    self.mlp_ratio = mlp_ratio
    self.num_heads = num_heads
    self.patch_size = patch_size
    self.num_register_tokens = num_register_tokens
    self.interpolate_antialias = interpolate_antialias
    self.interpolate_offset = interpolate_offset

    self.protein_patch_embed = embed_layer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        embed_dim=embed_dim)
    self.contour_patch_embed = embed_layer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=2,
        embed_dim=self.contour_embed_dim)
    num_patches = self.protein_patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
    self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, decoder_embed_dim))

    assert num_register_tokens >= 0
    self.register_tokens = (
        nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        if num_register_tokens else None
    )

    if drop_path_uniform is True:
      dpr = [drop_path_rate] * depth
      dpr_decoder = [drop_path_rate] * decoder_depth
    else:
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
      dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]

    common_block_kwargs = dict(
        block_name=block_name,
        ffn_layer=ffn_layer,
        act_layer=act_layer,
        norm_layer=norm_layer,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        ffn_bias=ffn_bias,
        init_values=init_values,
    )
    self.chunked_blocks = block_chunks > 0
    # encoder
    self.protein_encoder_blocks = create_blocks(
        block_chunks,
        depth,
        dim=embed_dim,
        num_heads=num_heads,
        drop_paths=dpr,
        **common_block_kwargs,
    )
    self.contour_encoder_blocks = create_blocks(
        block_chunks,
        depth,
        dim=self.contour_embed_dim,
        num_heads=num_heads,
        drop_paths=dpr,
        layer_idx_offset=depth,
        **common_block_kwargs,
    )
    self.encoder_norm = norm_layer(embed_dim)

    if generation_mode:
      # intergrater
      self.integrater_embed = nn.Linear(embed_dim, embed_dim, bias=bias)
      self.integrater_blocks = create_blocks(
          block_chunks,
          depth,
          dim=embed_dim,
          num_heads=num_heads,
          drop_paths=dpr,
          **common_block_kwargs,
      )
      self.integrater_norm = norm_layer(embed_dim)

    if reconstruction_mode or generation_mode:
      # decoder
      self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=bias)
      self.decoder_norm = norm_layer(decoder_embed_dim)
      self.decoder_blocks = create_blocks(
          block_chunks,
          decoder_depth,
          dim=decoder_embed_dim,
          num_heads=decoder_num_heads,
          drop_paths=dpr_decoder,
          **common_block_kwargs,
      )
      self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=bias)

    self.protein_mask_token = nn.Parameter(torch.zeros(1, embed_dim))
    self.contour_mask_token = nn.Parameter(torch.zeros(1, self.contour_embed_dim))
    self.init_weights()

  def init_weights(self):
    trunc_normal_(self.pos_embed, std=0.02)
    trunc_normal_(self.decoder_pos_embed, std=0.02)
    nn.init.normal_(self.cls_token, std=1e-6)
    if self.register_tokens is not None:
        nn.init.normal_(self.register_tokens, std=1e-6)
    if self.block_name in ['mamba1', 'mamba2']:
        named_apply(partial(
            init_weights_ssm,
            n_blocks=self.n_blocks,
            n_decoder_blocks=self.n_decoder_blocks,
            n_residuals_per_layer=int(self.mlp_ratio > 0) + 1),
        self)
    else:
      named_apply(init_weights_vit_timm, self)

  def unpatchify(self, patch_tokens):
    B, N, C = patch_tokens.shape
    h = w = int(N**.5)
    p = self.patch_size
    c = C // (p * p)
    x = patch_tokens.reshape(shape=(B, h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

  def interpolate_pos_encoding(self, x, w, h, pos_embed, has_cls_token=True):
    previous_dtype = x.dtype
    npatch = x.shape[1] - self.num_tokens
    N = pos_embed.shape[1] - self.num_tokens
    dim = x.shape[-1]
    patch_pos_embed = pos_embed[:, self.num_tokens:, :dim]
    if npatch == N and w == h:
      return (pos_embed if has_cls_token else patch_pos_embed).to(previous_dtype)
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M, f'input feature has wrong size, {N} is not a square for {M}'
    kwargs = {}
    if self.interpolate_offset:
      # Historical kludge: add a small number to avoid floating point error
      # in the interpolation, see https://github.com/facebookresearch/dino/issues/8
      # Note: still needed for backward-compatibility, the underlying operators
      # are using both output size and scale factors
      sx = float(w0 + self.interpolate_offset) / M
      sy = float(h0 + self.interpolate_offset) / M
      kwargs['scale_factor'] = (sx, sy)
    else:
      # Simply specify an output size instead of a scale factor
      kwargs['size'] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode='bicubic',
        antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    if has_cls_token:
      class_pos_embed = pos_embed[:, :self.num_tokens]
      return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)
    return patch_pos_embed.to(previous_dtype)

  def apply_feature_fusion(self, p, c):
    M = self.num_tokens + self.num_register_tokens
    if isinstance(p, list):
      shapes = [x.shape for x in p]
      cls_token = [x[:, :M] for x in p]
      patch_token = [x[:, M:].flatten(0, 1) for x in p]
      patch_token = torch.cat(patch_token, dim=0).unsqueeze(0)
      c = torch.cat([x.flatten(0, 1) for x in c], dim=0).unsqueeze(0)
    else:
      cls_token, patch_token = torch.tensor_split(p, [M], dim=1)
    # split the pos_token into c and p
    p_of_c, p_of_p = torch.tensor_split(patch_token, [self.contour_embed_dim], dim=2)
    # fusion the c into p_of_c
    # p_of_c = 0.5 * (p_of_c + c)
    ratio = torch.tanh(torch.mean(p_of_p, dim=2, keepdim=True))
    p_of_c = p_of_c + ratio * c
    p_of_c_mean = torch.mean(p_of_c, dim=2, keepdim=True)
    p_of_p = p_of_p - ratio * p_of_c_mean
    pos_token = torch.cat([p_of_c, p_of_p], dim=-1)
    # get the final p and c
    if isinstance(p, list):
      l = [s[0] * (s[1] - M) for s in shapes[:-1]]
      patch_token = torch.tensor_split(pos_token.squeeze(0), l, dim=0)
      p = [
          torch.cat([cls, p.view(s[0], s[1] - M, s[2])], dim=1).contiguous()
          for cls, p, s in zip(cls_token, patch_token, shapes)
      ]
    else:
      p = torch.cat([cls_token, pos_token], dim=1).contiguous()
    return p

  def encode(self, px, cx):
    for pblk, cblk in zip(self.protein_encoder_blocks, self.contour_encoder_blocks):
      px = pblk(px)
      cx = cblk(cx)
      px = self.apply_feature_fusion(px, cx)
    return px

  def integrate(self, feature, contour):
    B, nc, w, h = contour.shape
    feature = self.integrater_embed(feature)
    feature = feature + self.interpolate_pos_encoding(feature, w, h, self.pos_embed)
    contour = self.contour_patch_embed(contour)
    contour = contour + self.interpolate_pos_encoding(contour, w, h, self.pos_embed, False)
    for iblk, cblk in zip(self.integrater_blocks, self.contour_encoder_blocks):
      feature = iblk(feature)
      contour = cblk(contour)
      feature = self.apply_feature_fusion(feature, contour)
    feature = self.integrater_norm(feature)
    return feature

  def decode(self, x):
    x = self.decoder_embed(x)
    w = int(math.sqrt(x.shape[1] - self.num_tokens - self.num_register_tokens))
    h = w
    x = x + self.interpolate_pos_encoding(x, w, h, self.decoder_pos_embed)
    for blk in self.decoder_blocks:
      x = blk(x)
    x = self.decoder_norm(x)
    x = self.decoder_pred(x)
    return x[:, self.num_register_tokens + self.num_tokens:]

  def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    px = self.protein_patch_embed(x[:, :1])
    cx = self.contour_patch_embed(x[:, 1:])
    if masks is not None:
      px = torch.where(masks.unsqueeze(-1), self.protein_mask_token.to(px.dtype).unsqueeze(0), px)
      cx = torch.where(masks.unsqueeze(-1), self.contour_mask_token.to(cx.dtype).unsqueeze(0), cx)

    px = torch.cat((self.cls_token.expand(px.shape[0], -1, -1), px), dim=1)
    px = px + self.interpolate_pos_encoding(px, w, h, self.pos_embed)
    cx = cx + self.interpolate_pos_encoding(cx, w, h, self.pos_embed, False)

    if self.register_tokens is not None:
      px = torch.cat(
          (
              px[:, :self.num_tokens],
              self.register_tokens.expand(px.shape[0], -1, -1),
              px[:, self.num_tokens:],
          ),
          dim=1,
      )

    return px.contiguous(), cx.contiguous()

  def forward_features_list(self, x_list, masks_list):
    x = [self.prepare_tokens_with_masks(xi, masks) for xi, masks in zip(x_list, masks_list)]
    px = [p for p, _ in x]
    cx = [c for _, c in x]
    x = self.encode(px, cx)

    all_x = x
    output = []
    for x, masks in zip(all_x, masks_list):
      x_norm = self.encoder_norm(x)
      output.append(
          {
              'x_norm_clstoken': x_norm[:, :self.num_tokens],
              'x_norm_regtokens': x_norm[:, self.num_tokens:self.num_register_tokens+self.num_tokens],
              'x_norm_patchtokens': x_norm[:, self.num_register_tokens+self.num_tokens:],
              'x_prenorm': x,
              'x_norm': x_norm,
              'masks': masks,
          }
      )
    return output

  def forward_features(self, x, masks=None, **_):
    if isinstance(x, list):
        return self.forward_features_list(x, masks)

    px, cx = self.prepare_tokens_with_masks(x, masks)
    x = self.encode(px, cx)
    x_norm = self.encoder_norm(x)
    return {
        'x_norm_clstoken': x_norm[:, :self.num_tokens],
        'x_norm_regtokens': x_norm[:, self.num_tokens:self.num_register_tokens+self.num_tokens],
        'x_norm_patchtokens': x_norm[:, self.num_register_tokens+self.num_tokens:],
        'x_prenorm': x,
        'x_norm': x_norm,
        'masks': masks,
    }

  def _get_intermediate_layers_not_chunked(
      self,
      x,
      n: int = 1,
      norm: bool = True,
      contain_integrated_feature: bool = False
  ):
    (px, cx), cx_features = self.prepare_tokens_with_masks(x), []
    output, total_block_len = {}, len(self.protein_encoder_blocks)
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    for i, (pblk, cblk) in enumerate(zip(self.protein_encoder_blocks, self.contour_encoder_blocks)):
      px = pblk(px)
      cx = cblk(cx)
      cx_features.append(cx)
      px = self.apply_feature_fusion(px, cx)
      if i in blocks_to_take:
        output[i] = self.encoder_norm(px) if norm else px
    if contain_integrated_feature:
      px = self.integrater_embed(self.encoder_norm(px))
      for iblk, cx in zip(self.integrater_blocks, cx_features):
        px = iblk(px)
        px = self.apply_feature_fusion(px, cx)
        if i in blocks_to_take:
          f = self.integrater_norm(px) if norm else px
          # cls_token = (output[i][:, :1] + f[:, :1]) / 2
          # pos_token = torch.cat([output[i][:, 1:], f[:, 1:]], dim=1)
          output[i] = torch.cat([output[i], f], dim=-1)
    if (len(output) != len(blocks_to_take)):
      raise RuntimeError(f'only {len(output)} / {len(blocks_to_take)} blocks found.')
    return [output[i] for i in blocks_to_take]

  def _get_intermediate_layers_chunked(
      self,
      x,
      n: int = 1,
      norm: bool = True,
      contain_integrated_feature: bool = False
  ):
    (px, cx), cx_features = self.prepare_tokens_with_masks(x), []
    output, i, total_block_len = {}, 0, len(self.protein_encoder_blocks[-1])
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    for pblks, cblks in zip(self.protein_encoder_blocks, self.contour_encoder_blocks):
      for pblk, cblk in zip(pblks[i:], cblks[i:]):
        px = pblk(px)
        cx = cblk(cx)
        cx_features.append(cx)
        px = self.apply_feature_fusion(px, cx)
        if i in blocks_to_take:
          output[i] = self.encoder_norm(px) if norm else px
        i += 1
    if contain_integrated_feature:
      i = 0
      for iblks in self.integrater_blocks:
        for iblk, cx in zip(iblks, cx_features):
          px = iblk(px)
          px = self.apply_feature_fusion(px, cx)
          if i in blocks_to_take:
            f = self.integrater_norm(px)
            cls_token = (output[i][:, :1] + f[:, :1]) / 2
            pos_token = torch.cat([output[i][:, 1:], f[:, 1:]], dim=1)
            output[i] = torch.cat([cls_token, pos_token], dim=1)
          i += 1
    if (len(output) != len(blocks_to_take)):
      raise RuntimeError(f'only {len(output)} / {len(blocks_to_take)} blocks found.')
    return [output[i] for i in blocks_to_take]

  def get_intermediate_layers(
      self,
      x: torch.Tensor,
      n: Union[int, Sequence] = 1,  # Layers or n last layers to take
      reshape: bool = False,
      return_class_token: bool = False,
      norm: bool = True,
      contain_integrated_feature: bool = False,
  ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
    if not self.generation_mode and contain_integrated_feature:
      raise RuntimeError('Integrated feature is only available in generation mode.')
    if self.chunked_blocks:
      outputs = self._get_intermediate_layers_chunked(x, n, norm, contain_integrated_feature)
    else:
      outputs = self._get_intermediate_layers_not_chunked(x, n, norm, contain_integrated_feature)
    class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, 1+self.num_register_tokens:, self.contour_embed_dim:] for out in outputs]
    if reshape:
      B, _, w, h = x.shape
      outputs = [
          out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
          for out in outputs
      ]
    if return_class_token:
      return tuple(zip(outputs, class_tokens))
    return tuple(outputs)

  def generate_from_output(self, output, contours=None, **_):
    if isinstance(contours, list) and any([c is not None for c in contours]):
      return [self.generate_from_output(o, c) for o, c in zip(output, contours)]
    elif isinstance(contours, Tensor): # treat the output and contours as tensor
      feature = output['x_norm']
      reconstruction = self.decode(feature)
      output['reconstruction'] = self.unpatchify(reconstruction)
      if self.generation_mode:
        target_feature = self.integrate(feature, contours)
        generation = self.decode(target_feature)
        output['generation'] = self.unpatchify(generation)
    return output

  def generate(self, image, contour):
    px, cx = self.prepare_tokens_with_masks(image)
    px = self.encode(px, cx)
    px = self.encoder_norm(px)
    px = self.integrate(px, contour)
    px = self.decode(px)
    image = self.unpatchify(px)
    return image

  def forward(self, *args, **kwargs):
    output = self.forward_features(*args, **kwargs)
    if self.generation_mode or self.reconstruction_mode:
      output = self.generate_from_output(output, **kwargs)
    return output


def init_weights_vit_timm(module: nn.Module, name: str = ''):
  """ViT weight initialization, original timm impl (for reproducibility)"""
  if isinstance(module, nn.Linear):
    trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
      nn.init.zeros_(module.bias)


# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L86
def init_weights_ssm(
    module: nn.Module,
    n_blocks: int,
    n_decoder_blocks: int,
    n_residuals_per_layer: int,
    name: str = ''
):
  if isinstance(module, nn.Linear):
    trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
      if not getattr(module.bias, '_no_reinit', False):
        nn.init.zeros_(module.bias)
  for childname, p in module.named_parameters():
    if childname in ['out_proj.weight', 'fc2.weight']:
      n_layer = n_decoder_blocks if 'decoder' in name else n_blocks
      nn.init.kaiming_uniform_(p, a=math.sqrt(5))
      with torch.no_grad():
        p /= math.sqrt(n_residuals_per_layer * n_layer)


def prot_small(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtNet(
      block_name='attention',
      patch_size=patch_size,
      embed_dim=384,
      depth=12,
      num_heads=6,
      decoder_embed_dim=256,
      decoder_num_heads=8,
      num_register_tokens=num_register_tokens,
      norm_layer = partial(nn.LayerNorm, eps=1e-6),
      **kwargs,
  )
  return model


def prot_base(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtNet(
      block_name='attention',
      patch_size=patch_size,
      embed_dim=768,
      depth=12,
      num_heads=12,
      num_register_tokens=num_register_tokens,
      norm_layer = partial(nn.LayerNorm, eps=1e-6),
      **kwargs,
  )
  return model


def prot_mamba_small(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtNet(
      block_name='mamba2',
      patch_size=patch_size,
      embed_dim=384,
      depth=12,
      num_heads=6,
      decoder_embed_dim=256,
      decoder_num_heads=6,
      bias=False,
      num_register_tokens=num_register_tokens,
      norm_layer = partial(RMSNorm, eps=1e-5),
      **kwargs,
  )
  return model


def prot_mamba_base(patch_size=4, num_register_tokens=0, **kwargs):
  model = ProtNet(
      block_name='mamba1',
      patch_size=patch_size,
      embed_dim=128,
      depth=12,
      num_heads=12,
      bias=False,
      num_register_tokens=num_register_tokens,
      norm_layer = partial(RMSNorm, eps=1e-5),
      **kwargs,
  )
  return model
