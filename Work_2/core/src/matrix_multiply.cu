#include "../includes/matrix_multiply.cuh"
#include <__clang_cuda_builtin_vars.h>

void matrixMultiply(const float* A, const float* B, float* C) {
  int width = 128;
  int blockCount = 4;
  int threadCount = 32;
  int memSize = width * width * sizeof(float);

  float *hostMatrixA, *hostMatrixB, *hostMatrixC;
  float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;

  cudaMalloc((void**)&deviceMatrixA, memSize);
  cudaMalloc((void**)&deviceMatrixB, memSize);
  cudaMalloc((void**)&deviceMatrixC, memSize);

  cudaMemcpy(deviceMatrixA, hostMatrixA, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrixB, hostMatrixB, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrixC, hostMatrixC, memSize, cudaMemcpyHostToDevice);

  dim3 grid(blockCount, blockCount, 1);
  dim3 block(threadCount, threadCount, 1);
  matrixMultiplyKernel<<<grid, block>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC);

  cudaMemcpy(hostMatrixC, deviceMatrixC, memSize, cudaMemcpyDeviceToHost);

  cudaFree(deviceMatrixA);
  cudaFree(deviceMatrixB);
  cudaFree(deviceMatrixC);
}

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C) {
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  int width = blockDim.x * gridDim.x;
  int linear = i * width + j;

  int sum = 0;
  for (int k = 0; k < width; ++k) {
    sum += A[i * width + k] * B[k * width + j];
  }

  C[linear] = sum;
}

void matrixMultiplyHost(const float* A, const float* B, float* C) {
  int width = 128;

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      int sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += A[i * width + k] * B[k * width + j];
      }
      C[i * width + j] = sum;
    }
  }
}
