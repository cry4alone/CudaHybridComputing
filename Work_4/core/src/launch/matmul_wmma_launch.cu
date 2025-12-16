#include "../../includes/kernels.cuh"
#include "../../includes/matrix_view.cuh"
#include <mma.h>

using namespace nvcuda;

__global__ void kernel_matmul_wmma(MatrixView A, MatrixView B, MatrixView C);

void launch_matmul_wmma(MatrixView a, MatrixView b, MatrixView c) {
    dim3 grid(c.cols() / 16, c.rows() / 16);
    dim3 block(32, 1, 1);
    kernel_matmul_wmma<<<grid, block, 0>>>(a, b, c);
}

