#pragma once

#include <cstddef>
#include <vector>

class Data {
 public:
  Data(size_t rows, size_t cols);
  ~Data();

  Data(const Data&) = delete;
  Data& operator=(const Data&) = delete;

  void fill(const std::vector<float>& host_data);
  [[nodiscard]] std::vector<float> to_host() const;

  [[nodiscard]] size_t rows() const {
    return rows_;
  }

  [[nodiscard]] size_t cols() const {
    return cols_;
  }

  [[nodiscard]] float* device_data() {
    return d_data;
  }

 private:
  float* d_data = nullptr;
  size_t rows_ = 0;
  size_t cols_ = 0;
};