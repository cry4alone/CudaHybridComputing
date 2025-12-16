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

static float reduce_on_host(const float* d_partial, int count) {
    std::vector<float> h(count);
    cudaMemcpy(h.data(), d_partial, sizeof(float) * count, cudaMemcpyDeviceToHost);
    float s = 0.0f;
    for (int i = 0; i < count; ++i) s += h[i];
    return s;
}

void launch_vecred_nobr(VectorView v, float* d_out) {
    const int block = 256;
    const int grid = (static_cast<int>(v.size()) + block - 1) / block;
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, sizeof(float) * grid);
    kernel_vecred_nobr_impl<<<grid, block, sizeof(float) * block>>>(v, d_partial);
    float sum = reduce_on_host(d_partial, grid);
    cudaMemcpy(d_out, &sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_partial);
}

void launch_vecred_br(VectorView v, float* d_out) {
    const int block = 256;
    const int grid = (static_cast<int>(v.size()) + block - 1) / block;
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, sizeof(float) * grid);
    int numWarps = (block + 31) / 32;
    kernel_vecred_br_impl<<<grid, block, sizeof(float) * numWarps>>>(v, d_partial);
    float sum = reduce_on_host(d_partial, grid);
    cudaMemcpy(d_out, &sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_partial);
}