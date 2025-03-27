import math
import logging
import os
from omegaconf import OmegaConf

import prot.distributed as distributed
from prot.utils import setup_logging
from prot.utils import utils


def load_config(config_name: str):
  config_filename = config_name + '.yaml'
  package_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  return OmegaConf.load(os.path.join(package_dir, 'configs', config_filename))


default_config = load_config('ssl_default_config')


def load_and_merge_config(config_name: str):
  config = OmegaConf.create(default_config)
  loaded_config = load_config(config_name)
  return OmegaConf.merge(config, loaded_config)


def apply_scaling_rules_to_cfg(cfg):  # to fix
  if cfg.optim.scaling_rule == 'sqrt_wrt_1024':
    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr
    cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
    logging.info(f'sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}')
  else:
    base_lr = cfg.optim.base_lr
    cfg.optim.lr = base_lr * cfg.train.batch_size_per_gpu * distributed.get_global_size() / 512.0
  return cfg


def write_config(cfg, output_dir, name='config.yaml'):
  logging.info(OmegaConf.to_yaml(cfg))
  saved_cfg_path = os.path.join(output_dir, name)
  with open(saved_cfg_path, "w") as f:
    OmegaConf.save(config=cfg, f=f)
  return saved_cfg_path


def get_cfg_from_args(args):
  args.output_dir = os.path.abspath(args.output_dir)
  opts = args.opts or []
  args.opts = opts + [f'train.output_dir={args.output_dir}']
  default_cfg = OmegaConf.create(default_config)
  cfg = OmegaConf.load(args.config_file)
  cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
  return cfg


def default_setup(args):
  distributed.enable(overwrite=True)
  seed = getattr(args, 'seed', 0)
  rank = distributed.get_global_rank()
  setup_logging(output=args.output_dir, level=logging.INFO)
  logger = logging.getLogger('prot')
  utils.fix_random_seeds(seed + rank)
  logger.info("git:\n  {}\n".format(utils.get_sha()))
  logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
  """
  Create configs and perform basic setups.
  """
  cfg = get_cfg_from_args(args)
  os.makedirs(args.output_dir, exist_ok=True)
  default_setup(args)
  apply_scaling_rules_to_cfg(cfg)
  write_config(cfg, args.output_dir)
  return cfg
