#include <random>
#include <vector>
#include "benchmark/benchmark.h"
#include "kernels.cuh"
#include "matrix.cuh"

void multiply_tiled(const Matrix& a, const Matrix& b, Matrix& c) {
  launch_matmul_shmem(a.view(), b.view(), c.view(), 16);
}

void multiply_wmma(const Matrix& a, const Matrix& b, Matrix& c) {
  launch_matmul_wmma(a.view(), b.view(), c.view());
}

struct TiledAlgo {
  static void multiply(const Matrix& a, const Matrix& b, Matrix& c) {
    multiply_tiled(a, b, c);
  }
};

struct WMMAAlgo {
  static void multiply(const Matrix& a, const Matrix& b, Matrix& c) {
    multiply_wmma(a, b, c);
  }
};

std::vector<float> generate_random_data(size_t size) {
  std::vector<float> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (size_t i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
  return data;
}

template <typename MatMulStrategy>
static void BM_GpuMatMul(benchmark::State& state) {
  const size_t n = state.range(0);

  std::vector<float> host_A_data(n * n, 1.0f);
  std::vector<float> host_B_data(n * n, 2.0f);

  Matrix A(host_A_data, n, n);
  Matrix B(host_B_data, n, n);
  Matrix C(n, n);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    cudaEventRecord(start);

    MatMulStrategy::multiply(A, B, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000.0f);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

BENCHMARK_TEMPLATE(BM_GpuMatMul, TiledAlgo)
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Name("BM_CudaMatMul/Tiled_16");

BENCHMARK_TEMPLATE(BM_GpuMatMul, WMMAAlgo)
  ->RangeMultiplier(2)
  ->Range(64, 1024)
  ->Unit(benchmark::kMillisecond)
  ->UseManualTime()
  ->Name("BM_CudaMatMul/WMMA");

BENCHMARK_MAIN();