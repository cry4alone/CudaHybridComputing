#pragma once

#include <cuda_runtime.h>
#include "matrix_view.cuh"

// Launches the shared-memory tiled matrix multiplication kernel:
// C[m x n] = A[m x k] * B[k x n]
// Requirements: A.cols() == B.rows(), A.rows() == C.rows(), B.cols() == C.cols()
void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, cudaStream_t stream = 0);
