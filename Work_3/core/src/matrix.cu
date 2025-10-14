#include <cassert>
#include "../includes/matrix.cuh"

__global__ void kernel_matmul_naive(MatrixView a, MatrixView b, MatrixView c);

Matrix operator*(const Matrix& a, const Matrix& b) {
  assert(a.cols() == b.rows());

  size_t m = a.rows();
  size_t n = b.cols();
  size_t k = a.cols();

  Matrix c(m, n);

  dim3 blockDim(32, 32);

  dim3 gridDim((unsigned int)(n + blockDim.x - 1) / blockDim.x,
      (unsigned int)(m + blockDim.y - 1) / blockDim.y);

  kernel_matmul_naive<<<gridDim, blockDim>>>(a.view(), b.view(), c.view());

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "cUDa error: %s\n", cudaGetErrorString(err));
  }

  return c;
}

Matrix::Matrix(size_t rows, size_t cols)
    : data_(std::make_shared<Data>(rows, cols))
    , rows_(rows)
    , cols_(cols) {}

Matrix::Matrix(const std::vector<float>& host_data, size_t rows, size_t cols)
    : Matrix(rows, cols) {
  data_->fill(host_data);
}

std::vector<float> Matrix::to_host() const {
    return data_->to_host();
}