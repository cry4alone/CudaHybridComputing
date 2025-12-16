#include <cuda_runtime.h>
#include "../../includes/kernels.cuh"

// Kernel (1): block-level reduction using shared memory, no warp shuffles
__global__ void kernel_vecred_nobr_impl(VectorView v, float* partial_sums) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = static_cast<unsigned>(v.size());
    float x = (idx < n) ? v(idx) : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    // classic tree reduction in shared memory
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}
