# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py
#   https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mlp.py


from typing import Callable, Optional
from torch import Tensor, nn


class Mlp(nn.Module):
  def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      drop: float = 0.0,
      bias: bool = True,
  ) -> None:
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop = nn.Dropout(drop)

  def forward(self, x: Tensor) -> Tensor:
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class GatedMlp(nn.Module):
  def __init__(
      self,
      in_features,
      hidden_features=None,
      out_features=None,
      act_layer=nn.SiLU,
      drop: float = 0.0,
      bias=False,
  ):
    super().__init__()
    out_features = out_features if out_features is not None else in_features
    self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
    self.act = act_layer()
    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    y = self.fc1(x)
    y, gate = y.chunk(2, dim=-1)
    y = y * self.act(gate)
    y = self.fc2(y)
    y = self.drop(y)
    return y
