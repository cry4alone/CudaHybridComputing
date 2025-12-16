#include <cuda_runtime.h>
#include <vector>
#include "../../includes/kernels.cuh"
#include "../../includes/vecred_utils.cuh"

__global__ void kernel_vecred_br_impl(VectorView v, float* partial_sums);

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
