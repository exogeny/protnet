import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
  """
  Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al.
  Spreading vectors for similarity search
  """

  def __init__(self):
    super().__init__()
    self.pdist = nn.PairwiseDistance(2, eps=1e-8)

  def pairwise_NNs_inner(self, x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
    # max inner prod -> min distance
    _, I = torch.max(dots, dim=1)  # noqa: E741
    return I

  def forward(self, student_output, eps=1e-6):
    """
    Args:
        student_output (BxD): backbone output of student
    """
    with torch.amp.autocast(enabled=False, device_type="cuda"):
      student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
      I = self.pairwise_NNs_inner(student_output)  # noqa: E741
      distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
      distances = torch.clamp(distances, min=eps)
      loss = -torch.log(distances).mean()
    return loss
