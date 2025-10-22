#include <cassert>
#include "../includes/matrix.cuh"

__global__ void kernel_matmul_naive(MatrixView A, MatrixView B, MatrixView C);

Matrix operator*(const Matrix& A, const Matrix& B) {
  assert(A.cols() == B.rows());

  size_t m = A.rows();
  size_t n = B.cols();
  size_t k = A.cols();

  Matrix C(m, n);

  dim3 blockDim(32, 32);

  dim3 gridDim((unsigned int)(n + blockDim.x - 1) / blockDim.x,
      (unsigned int)(m + blockDim.y - 1) / blockDim.y);

  kernel_matmul_naive<<<gridDim, blockDim>>>(A.view(), B.view(), C.view());

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  }

  return C;
}

Data::Data(size_t r, size_t c)
    : rows(r)
    , cols(c) {
  cudaMalloc(&d_data, rows * cols * sizeof(float));
}

Data::~Data() {
  if (d_data) {
    cudaFree(d_data);
  }
}

void Data::fill(const std::vector<float>& host_data) {
  assert(host_data.size() == rows * cols);
  cudaMemcpy(
      d_data, host_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<float> Data::to_host() const {
  std::vector<float> host(rows * cols);
  cudaMemcpy(host.data(), d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return host;
}

MatrixView Matrix::view() const {
  // pitch = cols (row-major, no padding)
  return {data_->d_data, rows_, cols_, cols_};
}

Matrix::Matrix(size_t rows, size_t cols)
    : data_(std::make_shared<Data>(rows, cols))
    , rows_(rows)
    , cols_(cols) {}

Matrix::Matrix(const std::vector<float>& host_data, size_t rows, size_t cols)
    : Matrix(rows, cols) {
  data_->fill(host_data);
}