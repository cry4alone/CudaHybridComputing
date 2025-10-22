#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "data.cuh"
#include "matrix_view.cuh"

class Matrix {
 public:
  Matrix(size_t rows, size_t cols);
  Matrix(const std::vector<float>& host_data, size_t rows, size_t cols);

  [[nodiscard]] size_t rows() const {
    return rows_;
  }

  [[nodiscard]] size_t cols() const {
    return cols_;
  }

  [[nodiscard]] std::vector<float> to_host() const;
  [[nodiscard]] MatrixView view() const;

  Matrix(Matrix&&) = default;
  Matrix& operator=(Matrix&&) = default;
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  friend Matrix operator*(const Matrix& A, const Matrix& B);

 private:
  std::shared_ptr<Data> data_;
  size_t rows_, cols_;
};