import triton
import triton.language as tl


@triton.jit
def safe_sigmoid(x):
  dtype = x.dtype
  return tl.sigmoid(x.to(tl.float32)).to(dtype)
