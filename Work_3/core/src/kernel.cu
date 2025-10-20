#include "../includes/matrix_view.cuh"
#include "../includes/kernels.cuh"

// Fallback naive kernel (kept for reference/testing)
__global__ void kernel_matmul_naive(MatrixView a, MatrixView b, MatrixView c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < (int)c.rows() && col < (int)c.cols()) {
    float sum = 0.0f;
    for (int kk = 0; kk < (int)a.cols(); ++kk) {
      sum += a(row, kk) * b(kk, col);
    }
    c(row, col) = sum;
  }
}

// Shared-memory tiled kernel
template <int TILE>
__global__ void kernel_matmul_shmem(MatrixView A, MatrixView B, MatrixView C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  const int M = (int)C.rows();
  const int N = (int)C.cols();
  const int K = (int)A.cols();

  float acc = 0.0f;

  // Iterate over tiles of K dimension
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;

    // Load A tile element if in bounds
    if (row < M && a_col < K) {
      As[threadIdx.y][threadIdx.x] = A(row, a_col);
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load B tile element if in bounds
    if (b_row < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B(b_row, col);
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial accumulation for this tile
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C(row, col) = acc;
  }
}

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, cudaStream_t stream) {
  const int TILE = 16; // reasonable default for many GPUs
  dim3 block(TILE, TILE);
  dim3 grid((unsigned int)((c.cols() + TILE - 1) / TILE),
            (unsigned int)((c.rows() + TILE - 1) / TILE));

  // Dispatch based on TILE constant
  switch (TILE) {
    case 16:
      kernel_matmul_shmem<16><<<grid, block, 0, stream>>>(a, b, c);
      break;
    case 32:
      kernel_matmul_shmem<32><<<grid, block, 0, stream>>>(a, b, c);
      break;
    default:
      kernel_matmul_shmem<16><<<grid, block, 0, stream>>>(a, b, c);
      break;
  }
}