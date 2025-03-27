#include "selective_scan_fwd_kernel.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
