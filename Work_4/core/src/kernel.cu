#include <cuda_fp16.h>
#include <mma.h>
#include "../includes/kernels.cuh"
#include "../includes/matrix_view.cuh"

using namespace nvcuda;

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

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, int tile) {
  dim3 block(tile, tile);
  dim3 grid((c.cols() + tile - 1) / tile, (c.rows() + tile - 1) / tile);

  size_t shared_mem_size = 2 * tile * tile * sizeof(float);
  kernel_matmul_shmem<<<grid, block, shared_mem_size>>>(a, b, c, tile);
}

__global__ void kernel_matmul_wmma(MatrixView A, MatrixView B, MatrixView C) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  __shared__ __half shmem_a[16 * 16];
  __shared__ __half shmem_b[16 * 16];
  __shared__ float shmem_c[16 * 16];

  const int M = C.rows();
  const int N = C.cols();
  const int K = A.cols();

  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      int r = idx / 16;
      int c = idx % 16;
      int a_r = tile_m * 16 + r;
      int a_c = k0 + c;
      int b_r = k0 + r;
      int b_c = tile_n * 16 + c;

      float a_val = (a_r < M && a_c < K) ? A(a_r, a_c) : 0.0f;
      float b_val = (b_r < K && b_c < N) ? B(b_r, b_c) : 0.0f;
      shmem_a[idx] = __float2half(a_val);
      shmem_b[idx] = __float2half(b_val);
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, shmem_a, 16);
    wmma::load_matrix_sync(b_frag, shmem_b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(shmem_c, c_frag, 16, wmma::mem_row_major);
  __syncthreads();
  for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
    int r = idx / 16;
    int c = idx % 16;
    int g_r = tile_m * 16 + r;
    int g_c = tile_n * 16 + c;
    if (g_r < M && g_c < N) {
      C(g_r, g_c) = shmem_c[idx];
    }
  }
}

void launch_matmul_wmma(MatrixView a, MatrixView b, MatrixView c) {
  dim3 grid(c.cols() / 16, c.rows() / 16);
  dim3 block(32, 1, 1);
  kernel_matmul_wmma<<<grid, block>>>(a, b, c);
}