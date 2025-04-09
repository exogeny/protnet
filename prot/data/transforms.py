from typing import Sequence

import torch
from torchvision import transforms
from torchvision.transforms import _functional_pil as _FP


class GaussianBlur(transforms.RandomApply):
  """
  Apply Gaussian Blur to the PIL image.
  """

  def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
    # NOTE: torchvision is applying 1 - probability to return the original image
    keep_p = 1 - p
    transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
    super().__init__(transforms=[transform], p=keep_p)


class RandomAffine:
  def __init__(self, offset):
    self._offset = offset

  def __call__(self, pic):
    m1 = (torch.rand(2, 2) * 1.2 - 0.6) * (1 - torch.eye(2))
    m2 = (torch.rand(2, 2) * 2.7 + 0.3) * torch.eye(2)
    matrix = torch.linalg.inv(m1 + m2)
    trans = torch.rand(1, 2) * self._offset - self._offset / 2
    matrix = torch.cat([matrix, trans], dim=0).permute(1, 0)
    matrix = matrix.flatten().tolist()
    return _FP.affine(pic, matrix=matrix)

class MaybeToTensor(transforms.ToTensor):
  """
  Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
  """

  def __call__(self, pic):
    """
    Args:
        pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, torch.Tensor):
      return pic
    return super().__call__(pic)


DEFAULT_MEAN = (0.0695, 0.0760, 0.6815)
DEFAULT_STD = (0.0917, 0.1299, 0.4264)


def make_normalize_transform(
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Normalize:
  return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
):
  transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
  if hflip_prob > 0.0:
    transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
  transforms_list.extend(
      [
          MaybeToTensor(),
          make_normalize_transform(mean=mean, std=std),
      ]
  )
  return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
  transforms_list = [
      transforms.Resize(resize_size, interpolation=interpolation),
      transforms.CenterCrop(crop_size),
      MaybeToTensor(),
      make_normalize_transform(mean=mean, std=std),
  ]
  return transforms.Compose(transforms_list)
