#include "benchmark/benchmark.h"
#include "matrix.cuh"
#include <Eigen/Dense>
#include <vector>
#include <random>

// Функция для генерации случайных данных
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

// Бенчмарк для Eigen (CPU)
static void BM_EigenMatMul(benchmark::State& state) {
    const size_t n = state.range(0);

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf C(n, n);

    for (auto _ : state) {
        C = A * B;
        benchmark::DoNotOptimize(C.data());
    }
}
BENCHMARK(BM_EigenMatMul)->RangeMultiplier(2)->Range(16, 1024)->Unit(benchmark::kMillisecond);

// Бенчмарк для CUDA (GPU)
static void BM_CudaMatMul(benchmark::State& state) {
    const size_t n = state.range(0);

    // Подготовка данных на хосте
    auto host_A_data = generate_random_data(n * n);
    auto host_B_data = generate_random_data(n * n);

    // Выделение памяти на GPU и копирование (вне замера)
    Matrix A(host_A_data, n, n);
    Matrix B(host_B_data, n, n);
    Matrix C(n, n); // reuse output buffer across iterations

    // CUDA Events для замера времени на GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);

        // Only time the kernel; operator* synchronizes, but to ensure fairness we can launch and sync explicitly here by reusing * operator
        C = A * B;

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0f);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
BENCHMARK(BM_CudaMatMul)->RangeMultiplier(2)->Range(16, 1024)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_MAIN();