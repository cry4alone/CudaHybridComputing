#pragma once

#include <cuda_runtime.h>
#include "matrix_view.cuh"

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, int TILE);
__global__ void kernel_matmul_naive(MatrixView A, MatrixView B, MatrixView C);

void launch_matmul_naive(
    MatrixView a, MatrixView b, MatrixView c, cudaStream_t stream = 0);
