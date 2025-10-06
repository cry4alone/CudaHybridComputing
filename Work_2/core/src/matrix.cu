#include "../includes/matrix.cuh"
#include <cassert>

Data::Data(size_t r, size_t c) : rows(r), cols(c) {
    cudaMalloc(&d_data, rows * cols * sizeof(float));
}

Data::~Data() {
    if (d_data) {
        cudaFree(d_data);
    }
}

void Data::fill(const std::vector<float>& host_data) {
    assert(host_data.size() == rows * cols);
    cudaMemcpy(d_data, host_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
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
    : data_(std::make_shared<Data>(rows, cols)), rows_(rows), cols_(cols) {}

Matrix::Matrix(const std::vector<float>& host_data, size_t rows, size_t cols)
    : Matrix(rows, cols) {
    data_->fill(host_data);
}