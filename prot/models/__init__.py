from prot.models import vit as vits
from prot.models import prot as prots


def build_model(
    args,
    only_teacher=False,
    img_size=224,
    reconstruction_mode=False,
    generation_mode=False,
):
  args.arch = args.arch.removesuffix("_memeff")
  if "vit" in args.arch:
      vit_kwargs = dict(
          img_size=img_size,
          patch_size=args.patch_size,
          init_values=args.layerscale,
          ffn_layer=args.ffn_layer,
          block_chunks=args.block_chunks,
          qkv_bias=args.qkv_bias,
          proj_bias=args.proj_bias,
          ffn_bias=args.ffn_bias,
          num_register_tokens=args.num_register_tokens,
          interpolate_offset=args.interpolate_offset,
          interpolate_antialias=args.interpolate_antialias,
      )
      teacher = vits.__dict__[args.arch](**vit_kwargs)
      if only_teacher:
          return teacher, teacher.embed_dim
      student = vits.__dict__[args.arch](
          **vit_kwargs,
          drop_path_rate=args.drop_path_rate,
          drop_path_uniform=args.drop_path_uniform,
      )
      embed_dim = student.embed_dim
  if 'prot' in args.arch:
      vit_kwargs = dict(
          img_size=img_size,
          patch_size=args.patch_size,
          mlp_ratio=args.mlp_ratio,
          init_values=args.layerscale,
          ffn_layer=args.ffn_layer,
          block_chunks=args.block_chunks,
          qkv_bias=args.qkv_bias,
          proj_bias=args.proj_bias,
          ffn_bias=args.ffn_bias,
          num_tokens=args.num_tokens,
          num_register_tokens=args.num_register_tokens,
          interpolate_offset=args.interpolate_offset,
          interpolate_antialias=args.interpolate_antialias,
          reconstruction_mode=reconstruction_mode,
          generation_mode=generation_mode,
      )
      teacher = prots.__dict__[args.arch](**vit_kwargs)
      if only_teacher:
          return teacher, teacher.embed_dim
      student = prots.__dict__[args.arch](
          **vit_kwargs,
          drop_path_rate=args.drop_path_rate,
          drop_path_uniform=args.drop_path_uniform,
      )
      embed_dim = student.embed_dim
  return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
        reconstruction_mode=cfg.train.reconstruction,
        generation_mode=cfg.train.generation,
    )
