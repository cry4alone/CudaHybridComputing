#pragma once

#include <cuda_runtime.h>
#include "matrix_view.cuh"

void launch_matmul_shmem(MatrixView a, MatrixView b, MatrixView c, 
                         int TILE);