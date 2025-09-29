#ifndef WORK2_MATRIX_MULTIPLY_H
#define WORK2_MARTIX_MULTIPLY_H

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C);

void matrixMultiply(const float* A, const float* B, float* C);

void matrixMultiplyHost(const float* A, const float* B, float* C);

#endif // WORK2_MATRIX_MULTIPLY_H