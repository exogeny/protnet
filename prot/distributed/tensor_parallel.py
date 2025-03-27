from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from prot.utils.torch import custom_bwd, custom_fwd


from prot.distributed.distributed_utils import (
    all_gather_raw,
    all_reduce,
    all_reduce_raw,
    reduce_scatter,
    reduce_scatter_raw,
)


class ParallelLinearFunc(torch.autograd.Function):
  @staticmethod
  @custom_fwd
  def forward(ctx, x, weight, bias, process_group=None, sequence_parallel=True):
    """
    If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
    with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
    """
    ctx.compute_weight_gradient = weight.requires_grad
    ctx.process_group = process_group
    ctx.sequence_parallel = sequence_parallel

    if torch.is_autocast_enabled():
      x = x.to(dtype=torch.get_autocast_gpu_dtype())
    x = x.contiguous()
    if process_group is not None and sequence_parallel:
      # We want to kick off the all_gather early, before weight dtype conversion
      total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
    else:
      total_x = x

    if torch.is_autocast_enabled():
      weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
      bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
    weight = weight.contiguous()
    if process_group is not None and sequence_parallel:
        handle_x.wait()
    # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
    output = F.linear(total_x, weight, bias)
    if ctx.compute_weight_gradient:
      ctx.save_for_backward(x, weight)
    else:
      ctx.save_for_backward(weight)
    return output

  @staticmethod
  @custom_bwd
  def backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    process_group = ctx.process_group
    sequence_parallel = ctx.sequence_parallel
    if ctx.compute_weight_gradient:
      x, weight = ctx.saved_tensors
      if process_group is not None and sequence_parallel:
        total_x, handle_x = all_gather_raw(x, process_group, async_op=True)
      else:
        total_x = x
    else:
      (weight,) = ctx.saved_tensors
      total_x = None
    batch_shape = grad_output.shape[:-1]
    batch_dim = batch_shape.numel()
    grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
    if ctx.needs_input_grad[0]:
      grad_input = F.linear(grad_output, weight.t())
      grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
      if process_group is not None:
        reduce_fn = reduce_scatter_raw if sequence_parallel else all_reduce_raw
        grad_input, handle_grad_input = reduce_fn(grad_input, process_group, async_op=True)
    else:
      grad_input = None
    if ctx.needs_input_grad[1]:
      assert ctx.compute_weight_gradient
      if process_group is not None and sequence_parallel:
          handle_x.wait()
      grad_weight = torch.einsum(
          "bo,bi->oi", grad_output, total_x.reshape(batch_dim, total_x.shape[-1])
      )
    else:
      grad_weight = None
    grad_bias = grad_output.sum(dim=0) if ctx.needs_input_grad[2] else None
    if process_group is not None and ctx.needs_input_grad[0]:
      handle_grad_input.wait()
    return grad_input, grad_weight, grad_bias, None, None


def parallel_linear_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
):
  return ParallelLinearFunc.apply(x, weight, bias, process_group, sequence_parallel)


class ColumnParallelLinear(nn.Linear):
  def __init__(
      self,
      in_features: int,
      out_features: int,
      process_group: ProcessGroup,
      bias: bool = True,
      sequence_parallel=True,
      multiple_of=1,
      device=None,
      dtype=None,
  ) -> None:
    world_size = torch.distributed.get_world_size(process_group)
    if out_features % multiple_of:
        raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
    multiple = out_features // multiple_of
    # We want to split @multiple across world_size, but it could be an uneven split
    div = multiple // world_size
    mod = multiple % world_size
    # The first @mod ranks get @div + 1 copies, the rest get @div copies
    local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
    super().__init__(
        in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype
    )
    self.process_group = process_group
    self.sequence_parallel = sequence_parallel

  def forward(self, x):
    # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
    # we do an all_gather of x before doing the matmul.
    # If not, then the input is already gathered.
    return parallel_linear_func(
        x,
        self.weight,
        self.bias,
        process_group=self.process_group,
        sequence_parallel=self.sequence_parallel,
    )


class RowParallelLinear(nn.Linear):
  def __init__(
      self,
      in_features: int,
      out_features: int,
      process_group: ProcessGroup,
      bias: bool = True,
      sequence_parallel=True,
      multiple_of=1,
      device=None,
      dtype=None,
  ) -> None:
    world_size = torch.distributed.get_world_size(process_group)
    rank = torch.distributed.get_rank(process_group)
    if in_features % multiple_of:
        raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
    multiple = in_features // multiple_of
    # We want to split @multiple across world_size, but it could be an uneven split
    div = multiple // world_size
    mod = multiple % world_size
    # The first @mod ranks get @div + 1 copies, the rest get @div copies
    local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
    # Only rank 0 will have bias
    super().__init__(
        local_multiple * multiple_of,
        out_features,
        bias=bias and rank == 0,
        device=device,
        dtype=dtype,
    )
    self.process_group = process_group
    self.sequence_parallel = sequence_parallel

  def forward(self, x):
    """
    We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
    a reduce_scatter of the result.
    """
    out = parallel_linear_func(x, self.weight, self.bias)
    reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
    return reduce_fn(out, self.process_group)
