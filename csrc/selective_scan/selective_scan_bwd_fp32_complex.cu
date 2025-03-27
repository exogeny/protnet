#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<float, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
