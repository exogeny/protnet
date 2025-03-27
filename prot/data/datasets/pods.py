from typing import Any, Callable, Optional
import os
import numpy as np

from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder


class SubcellLocRemapping:
  def __init__(self, root, classes):
    self.ensg_locs = {}
    with open(os.path.join(root, 'ensgs.txt'), 'r') as f:
      for line in f.readlines():
        ensg, locs = line.strip().split(',', maxsplit=1)
        locs = locs.strip().split(',')
        locs = np.array([float(l) for l in locs])
        self.ensg_locs[ensg] = locs
    self.classes = classes

  def __call__(self, target):
    return self.ensg_locs[self.classes[target]]

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.cell_line})"


class PodsDataset(ImageFolder):
  def __init__(
      self,
      root: str,
      split: str,
      remapping: bool = False,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None):
    image_folder = os.path.join(root, split)
    classes, _ = self.find_classes(image_folder)
    self._targets = np.array(list(range(len(classes))))
    self._remapping = remapping
    if bool(remapping):
      print(f'Using pods dataset with remapping!')
      self.targets = np.array(list(range(13)))
      remapping_transform = SubcellLocRemapping(root, classes)
      if target_transform is not None:
        target_transform = Compose([remapping_transform, target_transform])
      else:
        target_transform = remapping_transform
    super().__init__(
        image_folder,
        transform,
        target_transform
    )

  def get_targets(self):
    if self._remapping:
      return np.array(range(13), dtype=np.int32)
    return np.array(self._targets, dtype=np.int32)
