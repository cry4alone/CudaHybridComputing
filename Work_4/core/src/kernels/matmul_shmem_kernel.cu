#include "../../includes/matrix_view.cuh"

__global__ void kernel_matmul_shmem(MatrixView A, MatrixView B, MatrixView C, int tile) {
  extern __shared__ float shared_mem[];
  float* As = shared_mem;
  float* Bs = &shared_mem[tile * tile];

  const int row = blockIdx.y * tile + threadIdx.y;
  const int col = blockIdx.x * tile + threadIdx.x;

  const int M = C.rows();
  const int N = C.cols();
  const int K = A.cols();

  float acc = 0.0f;

  for (int t = 0; t < (K + tile - 1) / tile; ++t) {
    int a_col = t * tile + threadIdx.x;
    int b_row = t * tile + threadIdx.y;

    if (row < M && a_col < K) {
      As[threadIdx.y * tile + threadIdx.x] = A(row, a_col);
    } else {
      As[threadIdx.y * tile + threadIdx.x] = 0.0f;
    }

    if (b_row < K && col < N) {
      Bs[threadIdx.y * tile + threadIdx.x] = B(b_row, col);
    } else {
      Bs[threadIdx.y * tile + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < tile; ++k) {
      acc += As[threadIdx.y * tile + k] * Bs[k * tile + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C(row, col) = acc;
  }
}
