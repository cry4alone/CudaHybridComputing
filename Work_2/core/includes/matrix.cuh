// matrix.cuh
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

class MatrixView;

class Data {
public:
    float* d_data = nullptr;
    size_t rows = 0;
    size_t cols = 0;

    Data(size_t r, size_t c);
    ~Data();

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;

    void fill(const std::vector<float>& host_data);
    [[nodiscard]] std::vector<float> to_host() const;
};

class MatrixView {
public:
    float* data = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t pitch = 0;

    __host__ __device__
    MatrixView() = default;

    __host__ __device__
    MatrixView(float* ptr, size_t r, size_t c, size_t p)
        : data(ptr), rows(r), cols(c), pitch(p) {}

    __device__
    float& operator()(size_t i, size_t j) {
        return data[i * pitch + j];
    }

    __device__
    const float& operator()(size_t i, size_t j) const {
        return data[i * pitch + j];
    }
};

class Matrix {
private:
    std::shared_ptr<Data> data_;
    size_t rows_, cols_;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<float>& host_data, size_t rows, size_t cols);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    [[nodiscard]] std::vector<float> to_host() const { return data_->to_host();}

    MatrixView view() const;

    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    friend Matrix operator*(const Matrix& A, const Matrix& B);
};