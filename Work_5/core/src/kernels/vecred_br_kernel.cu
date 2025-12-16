#include <cuda_runtime.h>
#include "../../includes/kernels.cuh"

// Kernel (2): warp-level reduction using __shfl_down_sync, shared memory for warp sums
__global__ void kernel_vecred_br_impl(VectorView v, float* partial_sums) {
    extern __shared__ float warp_sums[];
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = static_cast<unsigned>(v.size());
    float val = (idx < n) ? v(idx) : 0.0f;

    unsigned mask = 0xffffffffu;
    // warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warpId = threadIdx.x >> 5; // /32
    if (lane == 0) {
        warp_sums[warpId] = val;
    }
    __syncthreads();

    // reduce warp sums by first warp
    float block_sum = 0.0f;
    if (warpId == 0) {
        float wval = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            wval += __shfl_down_sync(mask, wval, offset);
        }
        if (lane == 0) block_sum = wval;
    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}
