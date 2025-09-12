#include <cuda_runtime.h>
#include "../includes/vector_add.cuh"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vector_add_launch(const float* a, const float* b, float* c, int n) {
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  size_t size = n * sizeof(float);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}