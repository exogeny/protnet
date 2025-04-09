from typing import Any, Callable, Optional
import os
import numpy as np
from PIL import Image

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


class PodsGenerationDataset:
  def __init__(self, root: str, transform: Optional[Callable] = None, **kwargs):
    generations_filename = os.path.join(root, 'generations.txt')
    if not os.path.exists(generations_filename):
      raise RuntimeError(f"Generations file list [{generations_filename}] not found.")
    with open(generations_filename, 'r') as f:
      generations = f.readlines()
    self.generations = []
    for line in generations:
      line = line.strip()
      if len(line) > 0:
        self.generations.append(line)
    self.transform = transform

  def __getitem__(self, index: int):
    image1, image2 = self.generations[index].split(',')
    image1 = Image.open(image1).convert(mode='RGB')
    image2 = Image.open(image2).convert(mode='RGB')
    if self.transform is not None:
      image1 = self.transform(image1)
      image2 = self.transform(image2)
    return image1, image2

  def __len__(self) -> int:
    return len(self.generations)
