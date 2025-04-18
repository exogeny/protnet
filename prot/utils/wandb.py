import os
from typing import Optional
from prot import distributed

WANDB_AVAILABLE = False


def set_wandb_args_parser(parser):
  parser.add_argument_group('wandb', 'W & B host, user name and project setting')
  parser.add_argument('--wandb-host', type=str, default='https://api.wandb.ai',
                      help='W & B host')
  parser.add_argument('--wandb-api-key', type=str, default=None,
                      help='W & B user name')
  parser.add_argument('--wandb-project', type=str, default=None,
                      help='W & B project name')
  return parser


def init(
    args,
    project: Optional[str] = None,
    name: Optional[str] = None,
    config=None
):
  global WANDB_AVAILABLE
  if distributed.is_main_process():
    api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')
    if api_key is not None:
      try:
        import wandb
        WANDB_AVAILABLE = wandb.login(host=args.wandb_host, key=api_key)
        if WANDB_AVAILABLE:
          wandb.init(
              project=args.wandb_project or project,
              name=name,
              config=config,
          )
      except:
        WANDB_AVAILABLE = False


def log(*args, **kwargs):
  if WANDB_AVAILABLE:
    import wandb
    wandb.log(*args, **kwargs)

def histogram(name, data):
  if WANDB_AVAILABLE:
    import wandb
    wandb.log({name: wandb.Histogram(data)})
