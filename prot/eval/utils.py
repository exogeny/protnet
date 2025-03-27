import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection
from prot.utils.helpers import MetricLogger


logger = logging.getLogger('prot')


class ModelWithNormalize(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, samples):
    return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
  def __init__(self, feature_model, n_last_blocks, autocast_ctx):
    super().__init__()
    self.feature_model = feature_model
    self.feature_model.eval()
    self.n_last_blocks = n_last_blocks
    self.autocast_ctx = autocast_ctx

  def forward(self, images):
    with torch.inference_mode():
      with self.autocast_ctx():
        features = self.feature_model.get_intermediate_layers(
            images, self.n_last_blocks, return_class_token=True)
    return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
  model.eval()
  if criterion is not None:
    criterion.eval()

  for metric in metrics.values():
    metric = metric.to(device)

  metric_logger = MetricLogger(delimiter='  ')
  header = 'Test:'
  for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
    outputs = model(samples.to(device))
    targets = targets.to(device)

    if criterion is not None:
      loss = criterion(outputs, targets)
      metric_logger.update(loss=loss.item())

    for k, metric in metrics.items():
      metric_inputs = postprocessors[k](outputs, targets)
      metric.update(**metric_inputs)

  metric_logger.synchronize_between_processes()
  logger.info(f'Average stats: {metric_logger}')
  stats = {k: metric.compute() for k, metric in metrics.items()}
  metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
  return metric_logger_stats, stats
