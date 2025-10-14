#pragma once

class MatrixView {
 public:
  __host__ __device__ MatrixView() = default;

  __host__ __device__ MatrixView(float* ptr, size_t r, size_t c)
      : data_(ptr)
      , rows_(r)
      , cols_(c) {}

  [[nodiscard]] __host__ __device__ float* data() {
    return data_;
  }

  [[nodiscard]] __host__ __device__ const float* data() const {
    return data_;
  }

  [[nodiscard]] __host__ __device__ size_t rows() const {
    return rows_;
  }

  [[nodiscard]] __host__ __device__ size_t cols() const {
    return cols_;
  }

  [[nodiscard]] __host__ __device__ size_t pitch() const {
    return pitch_;
  }

  __device__ float& operator()(size_t i, size_t j) {
    return data_[i * pitch_ + j];
  }

  __device__ const float& operator()(size_t i, size_t j) const {
    return data_[i * pitch_ + j];
  }

 private:
  float* data_ = nullptr;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t pitch_ = cols_;
};