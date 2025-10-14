#include "../includes/matrix_view.cuh"

__global__ void kernel_matmul_naive(MatrixView a, MatrixView b, MatrixView c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < c.rows() && col < c.cols()) {
    float sum = 0.0f;
    for (size_t k = 0; k < a.cols(); ++k) {
      sum += a(row, k) * b(k, col);
    }
    c(row, col) = sum;
  }
}