#pragma once

#include <cuda_runtime.h>
#include "matrix_view.cuh"

class VectorView {
public:
	__host__ __device__ VectorView() = default;
	__host__ __device__ VectorView(float* ptr, size_t n) : data_(ptr), size_(n) {}
	[[nodiscard]] __host__ __device__ float* data() { return data_; }
	[[nodiscard]] __host__ __device__ const float* data() const { return data_; }
	[[nodiscard]] __host__ __device__ size_t size() const { return size_; }
	__device__ float& operator()(size_t i) { return data_[i]; }
	__device__ const float& operator()(size_t i) const { return data_[i]; }
private:
	float* data_ = nullptr;
	size_t size_ = 0;
};

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, int TILE);

void launch_matmul_wmma(MatrixView a, MatrixView b, MatrixView c);

// Work 5: vector reduction kernels
void launch_vecred_nobr(VectorView v, float* d_out);
void launch_vecred_br(VectorView v, float* d_out);
