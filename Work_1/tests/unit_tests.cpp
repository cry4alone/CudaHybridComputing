#include <gtest/gtest.h>
#include <cuda_runtime.h> 
#include <vector>
#include <random>
#include <numeric>

// Подключаем заголовочный файл
#include "vector_add.cuh"




// Максимально допустимая погрешность при сравнении чисел с плавающей точкой.
constexpr float MAX_ERROR = 1e-5f;

/**
 * @class VectorAddTest
 * @brief Тестовый класс (fixture), который автоматически управляет 
 * выделением и освобождением памяти GPU для каждого теста.
 */
class VectorAddTest : public ::testing::Test {
protected:
    // Этот метод вызывается автоматически после каждого теста для очистки ресурсов.
    void TearDown() override {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }

    // Вспомогательный метод для подготовки тестовых данных.
    void PrepareData(int size) {
        n = size;
        if (n == 0) return;

        size_t bytes = n * sizeof(float);

        // 1. Готовим данные на хосте (CPU)
        h_a.resize(n);
        h_b.resize(n);
        h_c_gpu_result.resize(n);

        // Заполняем векторы случайными значениями для реалистичного теста.
        std::mt19937 gen(1337); // Используем фиксированное начальное число (seed) для воспроизводимости тестов.
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < n; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
        }

        // 2. Выделяем память на устройстве (GPU)
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
    }

    // Основная логика: запускаем вычисления и проверяем результаты.
    void RunAndVerify() {
        if (n == 0) {
            SUCCEED(); // Тест с нулевым количеством элементов считается успешным.
            return;
        }

        size_t bytes = n * sizeof(float);

        // 1. Вычисляем эталонный результат на CPU
        std::vector<float> h_c_cpu_ref(n);
        vector_add_cpu(h_a.data(), h_b.data(), h_c_cpu_ref.data(), n);

        // 2. Копируем входные данные с хоста (CPU) на устройство (GPU)
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

        // 3. Запускаем GPU-версию функции
        vector_add(d_a, d_b, d_c, n);
        // Проверяем, не было ли ошибок при запуске ядра на GPU
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Ошибка запуска ядра";

        // 4. Копируем результат обратно с устройства (GPU) на хост (CPU)
        cudaMemcpy(h_c_gpu_result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        // 5. Сравниваем результат GPU с эталонным результатом CPU
        for (int i = 0; i < n; ++i) {
            // ASSERT_NEAR используется для сравнения чисел с плавающей точкой
            ASSERT_NEAR(h_c_cpu_ref[i], h_c_gpu_result[i], MAX_ERROR)
                << "Расхождение в элементе с индексом " << i;
        }
    }

    // Переменные-члены, доступные в каждом тесте
    int n = 0; // Размер векторов
    // Векторы в памяти хоста (CPU). 'h_' от "host"
    std::vector<float> h_a, h_b, h_c_gpu_result; 
    // Указатели на память устройства (GPU). 'd_' от "device"
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
};

// --- Набор тестов ---
// Каждый TEST_F - это отдельный сценарий, использующий наш класс VectorAddTest.

// Тест со стандартным размером вектора
TEST_F(VectorAddTest, HandlesStandardSize) {
    PrepareData(1024);
    RunAndVerify();
}

// Тест с вектором нулевого размера (граничный случай)
TEST_F(VectorAddTest, HandlesZeroSizeVector) {
    PrepareData(0);
    RunAndVerify();
}

// Тест с вектором из одного элемента (граничный случай)
TEST_F(VectorAddTest, HandlesSingleElementVector) {
    PrepareData(1);
    RunAndVerify();
}

// Тест с размером, который не делится нацело на стандартный размер блока CUDA (важно!)
TEST_F(VectorAddTest, HandlesNonDivisibleSize) {
    PrepareData(999);
    RunAndVerify();
}

// Тест с большим вектором (проверка производительности и стабильности)
TEST_F(VectorAddTest, HandlesLargeVector) {
    PrepareData(1 << 16); // 65536 элементов
    RunAndVerify();
}