#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
