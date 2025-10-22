#include <cassert>
#include <iostream>
#include "../includes/matrix.cuh"
#include "../includes/kernels.cuh"

Matrix operator*(const Matrix& a, const Matrix& b) {
  assert(a.cols() == b.rows());

  const size_t m = a.rows();
  const size_t n = b.cols();
  const size_t k = a.cols();
  (void)k; // silence unused when NDEBUG

  Matrix c(m, n);

  launch_matmul_shmem(a.view(), b.view(), c.view(), 16);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << '\n';
  }
  // Ensure kernel completion before returning to avoid use-after-free when Matrix goes out of scope in callers
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA sync error: " << cudaGetErrorString(err) << '\n';
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