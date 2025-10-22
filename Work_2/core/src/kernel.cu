#include "../includes/matrix.cuh"

__global__ void kernel_matmul_naive(MatrixView A, MatrixView B, MatrixView C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < C.rows && col < C.cols) {
    float sum = 0.0f;
    for (size_t k = 0; k < A.cols; ++k) {
      sum += A(row, k) * B(k, col);
    }
    C(row, col) = sum;
  }
}