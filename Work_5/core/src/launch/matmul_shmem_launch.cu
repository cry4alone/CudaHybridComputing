#include "../../includes/kernels.cuh"
#include "../../includes/matrix_view.cuh"

__global__ void kernel_matmul_shmem(MatrixView A, MatrixView B, MatrixView C, int TILE);

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, int TILE) {
  dim3 block(TILE, TILE);
  dim3 grid((c.cols() + TILE - 1) / TILE, (c.rows() + TILE - 1) / TILE);
  size_t shared_mem_size = 2 * TILE * TILE * sizeof(float);
  kernel_matmul_shmem<<<grid, block, shared_mem_size>>>(a, b, c, TILE);
}
