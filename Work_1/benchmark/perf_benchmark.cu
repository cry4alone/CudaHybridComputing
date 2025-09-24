#include "benchmark/benchmark.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "vector_add.cuh"

// CUDA kernel prototype (should match the definition in your .cu file)
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n);

// Структура для хранения данных между SetUp и замерами
struct GpuBenchmarkState {
    std::vector<float> a, b, c;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    int n = 0;

    ~GpuBenchmarkState() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }
};

// CPU Benchmark — оставляем как есть (он корректен)
static void cpu_vector_add(benchmark::State& state) {
    const int n = state.range(0);

    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    for (auto _ : state) {
        vector_add_cpu(a.data(), b.data(), c.data(), n);
        benchmark::ClobberMemory(); 
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 3 * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

// GPU Benchmark — ПЕРЕПИСЫВАЕМ
static void gpu_vector_add(benchmark::State& state) {
    const int n = state.range(0);
    GpuBenchmarkState gpu_state;
    gpu_state.n = n;

    // Генерация данных
    gpu_state.a.resize(n);
    gpu_state.b.resize(n);
    gpu_state.c.resize(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        gpu_state.a[i] = dist(gen);
        gpu_state.b[i] = dist(gen);
    }

    // Выделение памяти на GPU — ДО замеров
    size_t size = n * sizeof(float);
    cudaMalloc(&gpu_state.d_a, size);
    cudaMalloc(&gpu_state.d_b, size);
    cudaMalloc(&gpu_state.d_c, size);

    // Копирование данных на GPU — ДО замеров
    cudaMemcpy(gpu_state.d_a, gpu_state.a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_state.d_b, gpu_state.b.data(), size, cudaMemcpyHostToDevice);

    // Подготовка для замера времени ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        // Замеряем ТОЛЬКО время выполнения ядра
        cudaEventRecord(start);
        vector_add_kernel<<<gridSize, blockSize>>>(gpu_state.d_a, gpu_state.d_b, gpu_state.d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Ждём завершения ядра

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000.0); // в секундах
    }

    // Копируем результат обратно (не в замере)
    cudaMemcpy(gpu_state.c.data(), gpu_state.d_c, size, cudaMemcpyDeviceToHost);

    // Устанавливаем статистику
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 3 * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);

    // Очистка
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Регистрация бенчмарков
BENCHMARK(cpu_vector_add)
    ->ArgName("N")
    ->Arg(1 << 10)   // 1K
    ->Arg(1 << 15)   // 32K
    ->Arg(1 << 20)   // 1M
    ->Arg(1 << 24);  // 16M

BENCHMARK(gpu_vector_add)
    ->ArgName("N")
    ->Arg(1 << 10)   // 1K
    ->Arg(1 << 15)   // 32K
    ->Arg(1 << 20)   // 1M
    ->Arg(1 << 24)   // 16M
    ->UseManualTime(); // ← ВАЖНО: мы сами управляем временем

BENCHMARK_MAIN();