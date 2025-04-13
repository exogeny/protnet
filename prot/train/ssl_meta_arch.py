from functools import partial
import logging

import math
import torch
from torch import nn
from xformers.ops import fmha
from torchmetrics.image import StructuralSimilarityIndexMeasure

from prot.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from prot.models import build_model_from_cfg
from prot.layers import BlockChunk, DINOHead
from prot.utils.utils import has_batchnorms
from prot.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from prot.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model


logger = logging.getLogger('prot')


class SSLMetaArch(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

    student_model_dict = dict()
    teacher_model_dict = dict()

    student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
    student_model_dict['backbone'] = student_backbone
    teacher_model_dict['backbone'] = teacher_backbone
    logger.info(f'OPTIONS -- architecture : embed_dim: {embed_dim}')

    if cfg.student.pretrained_weights:
      chkpt = torch.load(cfg.student.pretrained_weights)
      logger.info(f'OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}')
      if 'model' in chkpt:
        state_dict = chkpt['model']
      elif 'student' in chkpt:
        state_dict = chkpt['student']
      else:
        raise ValueError(
            f'Checkpoint {cfg.student.pretrained_weights} does not contain "model" or "student" key.'
        )
      student_backbone.load_state_dict(state_dict, strict=False)

    self.num_tokens = cfg.student.num_tokens
    self.embed_dim = embed_dim
    self.dino_out_dim = cfg.dino.head_n_prototypes

    # configure loss weights
    self.do_dino = cfg.dino.loss_weight > 0
    self.do_koleo = cfg.dino.koleo_loss_weight > 0
    self.do_ibot = cfg.ibot.loss_weight > 0
    self.ibot_separate_head = cfg.ibot.separate_head

    logger.info('OPTIONS -- DINO')
    if self.do_dino:
      logger.info(f'OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}')
      logger.info(f'OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}')
      logger.info(f'OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}')
      logger.info(f'OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}')
      logger.info(f'OPTIONS -- DINO -- using mlp bias: {cfg.dino.head_mlp_bias}')
      self.dino_loss_weight = cfg.dino.loss_weight
      dino_head = partial(
          DINOHead,
          in_dim=embed_dim,
          out_dim=cfg.dino.head_n_prototypes,
          hidden_dim=cfg.dino.head_hidden_dim,
          bottleneck_dim=cfg.dino.head_bottleneck_dim,
          nlayers=cfg.dino.head_nlayers,
          mlp_bias=cfg.dino.head_mlp_bias,
      )
      self.dino_loss = DINOLoss(self.dino_out_dim, self.num_tokens)
      if self.do_koleo:
        logger.info('OPTIONS -- DINO -- applying KOLEO regularization')
        self.koleo_loss = KoLeoLoss()

    else:
      logger.info('OPTIONS -- DINO -- not using DINO')

    if self.do_dino or self.do_ibot:
      student_model_dict['dino_head'] = dino_head()
      teacher_model_dict['dino_head'] = dino_head()

    logger.info('OPTIONS -- IBOT')
    logger.info(f'OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}')
    logger.info(f'OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}')
    logger.info(f'OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}')
    if self.do_ibot:
      self.ibot_loss_weight = cfg.ibot.loss_weight
      assert max(cfg.ibot.mask_ratio_min_max) > 0, 'please provide a positive mask ratio tuple for ibot'
      assert cfg.ibot.mask_sample_probability > 0, 'please provide a positive mask probability for ibot'
      self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
      self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
      if self.ibot_separate_head:
        logger.info(f'OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}')
        logger.info(f'OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}')
        logger.info(f'OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}')
        logger.info(f'OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}')
        logger.info(f'OPTIONS -- IBOT -- using mlp bias: {cfg.ibot.head_mlp_bias}')
        ibot_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
            mlp_bias=cfg.ibot.head_mlp_bias,
        )
        student_model_dict['ibot_head'] = ibot_head()
        teacher_model_dict['ibot_head'] = ibot_head()
      else:
        logger.info('OPTIONS -- IBOT -- head shared with DINO')

    self.need_to_synchronize_fsdp_streams = True

    self.student = nn.ModuleDict(student_model_dict)
    self.teacher = nn.ModuleDict(teacher_model_dict)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in self.teacher.parameters():
      p.requires_grad = False
    logger.info(f'Student and Teacher are built: they are both {cfg.student.arch} network.')
    # calcuate the SSIM metrics
    self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

  def ssim(self, x, y):
    return self._ssim(x[:, :1], y[:, :1])

  def forward(self, inputs):
    raise NotImplementedError

  def backprop_loss(self, loss):
    if self.fp16_scaler is not None:
      self.fp16_scaler.scale(loss).backward()
    else:
      loss.backward()

  def forward_backward(self, images, teacher_temp):
    n_global_crops = 2
    assert n_global_crops == 2
    n_local_crops = self.cfg.crops.local_crops_number

    global_crops = images['collated_global_crops'].cuda(non_blocking=True)
    global_crops_normed = images['collated_global_crops_normed'].cuda(non_blocking=True)
    local_crops = images['collated_local_crops'].cuda(non_blocking=True)

    masks = images['collated_masks'].cuda(non_blocking=True)
    mask_indices_list = images['mask_indices_list'].cuda(non_blocking=True)
    n_masked_patches_tensor = images['n_masked_patches'].cuda(non_blocking=True)
    n_masked_patches = mask_indices_list.shape[0]
    upperbound = images['upperbound']
    masks_weight = images['masks_weight'].cuda(non_blocking=True)

    n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
    n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

    do_dino = self.do_dino
    do_ibot = self.do_ibot

    # loss scales
    ibot_loss_scale = 1.0 / n_global_crops

    # teacher output
    @torch.no_grad()
    def get_teacher_output():
      x, n_global_crops_teacher = global_crops_normed, n_global_crops
      teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
      teacher_cls_tokens = teacher_backbone_output_dict['x_norm_clstoken']
      teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
      # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
      teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]), dim=0)
      teacher_cls_tokens = teacher_cls_tokens.flatten(0, 1)
      ibot_teacher_patch_tokens = teacher_backbone_output_dict['x_norm_patchtokens']
      _dim = ibot_teacher_patch_tokens.shape[-1]
      n_cls_tokens = teacher_cls_tokens.shape[0]

      if do_ibot and not self.ibot_separate_head:
        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
        buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
            out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
        )
        tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
        teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
        masked_teacher_patch_tokens_after_head = tokens_after_head[
            n_cls_tokens : n_cls_tokens + n_masked_patches
        ]
      elif do_ibot and self.ibot_separate_head:
        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
            out=buffer_tensor_teacher[:n_masked_patches],
        )
        teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
        masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
            :n_masked_patches
        ]
      else:
        teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
        masked_teacher_ibot_softmaxed_centered = None

      teacher_cls_tokens_after_head = teacher_cls_tokens_after_head.unflatten(0, (-1, self.num_tokens))
      if self.cfg.train.centering == 'centering':
        teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
        ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
        self.dino_loss.update_center(teacher_cls_tokens_after_head)
        if do_ibot:
            masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
            masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
            )
            masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
            self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])
      elif self.cfg.train.centering == 'sinkhorn_knopp':
        teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head, teacher_temp=teacher_temp
        ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

        if do_ibot:
          masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
              masked_teacher_patch_tokens_after_head,
              teacher_temp=teacher_temp,
              n_masked_patches_tensor=n_masked_patches_tensor,
          )
      else:
          raise NotImplementedError
      return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
    reshard_fsdp_model(self.teacher)

    loss_dict = {}
    loss_accumulator = 0  # for backprop
    # reverse the order of the global crops for generation
    # we generate the image with range [0, 1] with global_crops
    images = global_crops.chunk(n_global_crops, dim=0)
    images = torch.cat([images[1], images[0]], dim=0)
    contours = global_crops_normed[:, 1:].chunk(n_global_crops, dim=0)
    contours = torch.cat([contours[1], contours[0]], dim=0)
    student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
        [global_crops_normed, local_crops],
        masks=[masks, None],
        contours=[contours, None],
        is_training=True
    )

    # 0: perform the reconstruction and generation mse loss
    if self.cfg.train.reconstruction or self.cfg.train.generation:
      B, _, H, W = global_crops.shape
      mse_weight = torch.tensor([10, 1, 1], dtype=global_crops.dtype, device=global_crops.device)
      mse_weight = mse_weight / mse_weight.sum()
      mse_weight = mse_weight.view(1, 3, 1, 1).repeat((B, 1, H, W))
    if self.cfg.train.reconstruction:
      reconstruction = student_global_backbone_output_dict['reconstruction']
      reconstruction_loss = nn.functional.mse_loss(reconstruction, global_crops, weight=mse_weight)
      loss_accumulator += reconstruction_loss
      loss_dict['reconstruction'] = reconstruction_loss
      loss_dict['ssim_r'] = self.ssim(reconstruction, global_crops)

    if self.cfg.train.generation:
      generation = student_global_backbone_output_dict['generation']
      generation_loss = nn.functional.mse_loss(generation, images, weight=mse_weight)
      loss_accumulator += generation_loss
      loss_dict['generation'] = generation_loss
      loss_dict['ssim_g'] = self.ssim(generation, images)

    inputs_for_student_head_list = []

    # 1a: local crops cls tokens
    student_local_cls_tokens = student_local_backbone_output_dict['x_norm_clstoken']
    student_local_cls_tokens = student_local_cls_tokens.flatten(0, 1)
    inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

    # 1b: global crops cls tokens
    student_global_cls_tokens = student_global_backbone_output_dict['x_norm_clstoken']
    student_global_cls_tokens = student_global_cls_tokens.flatten(0, 1)
    inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

    # 1c: global crops patch tokens
    if do_ibot:
      _dim = student_global_backbone_output_dict['x_norm_clstoken'].shape[-1]
      ibot_student_patch_tokens = student_global_backbone_output_dict['x_norm_patchtokens']
      buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
      buffer_tensor_patch_tokens[:n_masked_patches].copy_(
          torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
      )
      if not self.ibot_separate_head:
        inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
      else:
        student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
            :n_masked_patches
        ]

    # 2: run
    _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
    outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

    # 3a: local crops cls tokens
    student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0).unflatten(0, (-1, self.num_tokens))

    # 3b: global crops cls tokens
    student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0).unflatten(0, (-1, self.num_tokens))

    # 3c: global crops patch tokens
    if do_ibot and not self.ibot_separate_head:
      student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

    if n_local_crops > 0:
      dino_local_crops_loss = self.dino_loss(
          student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
          teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
      ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
      loss_dict['dino_local_crops_loss'] = dino_local_crops_loss
      loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

    # process global crops
    loss_scales = 2  # this is here since we process global crops together

    if do_dino:
      dino_global_crops_loss = (
          self.dino_loss(
              student_output_list=[student_global_cls_tokens_after_head],
              teacher_out_softmaxed_centered_list=[
                  teacher_dino_softmaxed_centered_list.flatten(0, 1)
              ],  # these were chunked and stacked in reverse so A is matched to B
          )
          * loss_scales
          / (n_global_crops_loss_terms + n_local_crops_loss_terms)
      )
      loss_dict['dino_global_crops_loss'] = dino_global_crops_loss
      loss_accumulator += self.dino_loss_weight * dino_global_crops_loss
      student_cls_tokens = student_global_cls_tokens

      if self.do_koleo:
        koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
            self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
        )  # we don't apply koleo loss between cls tokens of a same image
        loss_accumulator += koleo_loss
        loss_dict['koleo_loss'] = (
            koleo_loss / loss_scales
        )  # this is to display the same losses as before but we can remove eventually

    if do_ibot:
      ibot_patch_loss = (
          self.ibot_patch_loss.forward_masked(
              student_global_masked_patch_tokens_after_head,
              masked_teacher_ibot_softmaxed_centered,
              student_masks_flat=masks,
              n_masked_patches=n_masked_patches,
              masks_weight=masks_weight,
          )
          * loss_scales
          * ibot_loss_scale
      )
      loss_dict['ibot_loss'] = ibot_patch_loss / 2
      loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

    self.backprop_loss(loss_accumulator)
    self.fsdp_synchronize_streams()
    return loss_dict

  def fsdp_synchronize_streams(self):
    if self.need_to_synchronize_fsdp_streams:
      torch.cuda.synchronize()
      for attr in {
          '_unshard_stream',
          '_post_backward_stream',
          '_pre_unshard_stream',
          '_all_reduce_stream',
          '_default_stream'
      }:
        stream = getattr(self.teacher.backbone, attr)
        setattr(self.student.dino_head, attr, stream)
        setattr(self.teacher.dino_head, attr, stream)
        setattr(self.student.backbone, attr, stream)
    self.need_to_synchronize_fsdp_streams = False

  def update_teacher(self, m):
    student_param_list = []
    teacher_param_list = []
    with torch.no_grad():
      for k in self.student.keys():
        for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
          student_param_list += ms.params
          teacher_param_list += mt.params
      torch._foreach_mul_(teacher_param_list, m)
      torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

  def train(self):
    super().train()
    self.teacher.eval()

  def get_maybe_fused_params_for_submodel(self, m):
    params_groups = get_params_groups_with_decay(
        model=m,
        lr_decay_rate=self.cfg.optim.layerwise_decay,
        patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
    )
    fused_params_groups = fuse_params_groups(params_groups)
    logger.info('fusing param groups')

    for g in fused_params_groups:
      g['foreach'] = True
    return fused_params_groups

  def get_params_groups(self):
    all_params_groups = []
    for m in self.student.values():
      all_params_groups += self.get_maybe_fused_params_for_submodel(m)
    return all_params_groups

  def prepare_for_distributed_training(self):
    logger.info('DISTRIBUTED FSDP -- preparing model for distributed training')
    if has_batchnorms(self.student):
      raise NotImplementedError
    # below will synchronize all student subnetworks across gpus:
    for k, v in self.student.items():
      self.teacher[k].load_state_dict(self.student[k].state_dict())
      student_model_cfg = self.cfg.compute_precision.student[k]
      self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
      teacher_model_cfg = self.cfg.compute_precision.teacher[k]
      self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
