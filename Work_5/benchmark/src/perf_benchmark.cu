#include <random>
#include <vector>
#include "benchmark/benchmark.h"
#include "kernels.cuh"

static std::vector<float> make_data(size_t n) {
  std::vector<float> v(n);
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) v[i] = dis(gen);
  return v;
}

static void BM_VecRed_NoBR(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  auto h = make_data(n);
  float* d_vec = nullptr;
  cudaMalloc(&d_vec, sizeof(float) * n);
  cudaMemcpy(d_vec, h.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
  float* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(float));

  VectorView vv(d_vec, n);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    cudaEventRecord(start);
    launch_vecred_nobr(vv, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0f);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_out);
  cudaFree(d_vec);
}

static void BM_VecRed_BR(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  auto h = make_data(n);
  float* d_vec = nullptr;
  cudaMalloc(&d_vec, sizeof(float) * n);
  cudaMemcpy(d_vec, h.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
  float* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(float));

  VectorView vv(d_vec, n);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    cudaEventRecord(start);
    launch_vecred_br(vv, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    state.SetIterationTime(ms / 1000.0f);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_out);
  cudaFree(d_vec);
}

// n in {8*2^0, 8*2^1, ..., up to a reasonable size}
BENCHMARK(BM_VecRed_NoBR)
    ->RangeMultiplier(2)
    ->Range(8, 8 << 20)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime()
    ->Name("BM_VecRed/NoBR");

BENCHMARK(BM_VecRed_BR)
    ->RangeMultiplier(2)
    ->Range(8, 8 << 20)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime()
    ->Name("BM_VecRed/BR");

BENCHMARK_MAIN();