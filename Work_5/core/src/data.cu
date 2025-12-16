#include <cassert>
#include <stdexcept>
#include "../includes/data.cuh"

Data::Data(size_t r, size_t c)
    : rows_(r)
    , cols_(c) {
  const size_t num_bytes = rows_ * cols_ * sizeof(float);
  cudaError_t err = cudaMalloc(&d_data, num_bytes);
  if (err != cudaSuccess) {
    throw std::bad_alloc{};
  }
}

Data::~Data() {
  if (d_data) {
    cudaFree(d_data);
  }
}

void Data::fill(const std::vector<float>& host_data) {
  if (host_data.size() != rows_ * cols_) {
    throw std::invalid_argument("host_data size does not match matrix dimensions");
  }
  cudaError_t err = cudaMemcpy(
      d_data, host_data.data(), rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed in fill");
  }
}

std::vector<float> Data::to_host() const {
  std::vector<float> host(rows_ * cols_);
  cudaError_t err = cudaMemcpy(
      host.data(), d_data, rows_ * cols_ * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed in to_host");
  }
  return host;
}