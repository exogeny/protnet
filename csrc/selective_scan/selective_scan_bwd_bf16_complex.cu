#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<at::BFloat16, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
