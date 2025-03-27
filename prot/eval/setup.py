import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from prot.models import build_model_from_cfg
from prot.utils.config import setup
from prot.utils import utils


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
  parser = argparse.ArgumentParser(
      description=description,
      parents=parents,
      add_help=add_help,
  )
  parser.add_argument(
      "--config-file",
      type=str,
      help="Model configuration file",
  )
  parser.add_argument(
      "--pretrained-weights",
      type=str,
      help="Pretrained model weights",
  )
  parser.add_argument(
      "--output-dir",
      default="",
      type=str,
      help="Output directory to write results and logs",
  )
  parser.add_argument(
      "--opts",
      help="Extra configuration options",
      default=[],
      nargs="+",
  )
  return parser


def build_model_for_eval(config, pretrained_weights):
  model, _ = build_model_from_cfg(config, only_teacher=True)
  utils.load_pretrained_weights(model, pretrained_weights, 'teacher')
  model.eval()
  model.cuda()
  return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
  cudnn.benchmark = True
  config = setup(args)
  model = build_model_for_eval(config, args.pretrained_weights)
  autocast_dtype = utils.get_autocast_dtype(config)
  return model, autocast_dtype
