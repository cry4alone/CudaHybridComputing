#include "../includes/kernels.cuh"
#include "../includes/matrix_view.cuh"

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

void launch_matmul_naive(MatrixView a, MatrixView b, MatrixView c, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((unsigned int)((c.cols() + 15) / 16), (unsigned int)((c.rows() + 15) / 16));

  kernel_matmul_naive<<<grid, block, 0, stream>>>(a, b, c);
}

__global__ void kernel_matmul_shmem(MatrixView A, MatrixView B, MatrixView C, int TILE) {

  extern __shared__ float shared_mem[];
  float* As = shared_mem;
  float* Bs = &shared_mem[TILE * TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  const int M = C.rows();
  const int N = C.cols();
  const int K = A.cols();

  float acc = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;

    if (row < M && a_col < K) {
      As[threadIdx.y * TILE + threadIdx.x] = A(row, a_col);
    } else {
      As[threadIdx.y * TILE + threadIdx.x] = 0.0f;
    }

    if (b_row < K && col < N) {
      Bs[threadIdx.y * TILE + threadIdx.x] = B(b_row, col);
    } else {
      Bs[threadIdx.y * TILE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y * TILE + k] * Bs[k * TILE + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C(row, col) = acc;
  }
}

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, int TILE) {
  dim3 block(TILE, TILE);
  dim3 grid((c.cols() + TILE - 1) / TILE, (c.rows() + TILE - 1) / TILE);

  size_t shared_mem_size = 2 * TILE * TILE * sizeof(float);
  kernel_matmul_shmem<<<grid, block, shared_mem_size>>>(a, b, c, TILE);
}