#include "../includes/vector_add.h"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = threadIdx + blockIdx * dimIdx;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vector_add_launch(const float* a, const float* b, float* c, int n) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  vector_add_kernel<<<grid, block>>>(a, b, c, n);
  cudaDeviceSynchronize();
}