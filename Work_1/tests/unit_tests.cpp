#include <gtest/gtest.h>
#include <cuda_runtime.h> 
#include <gtest/gtest.h>
#include <vector>
#include <cstdlib> // Для rand()
#include <ctime>   // Для time()
#include "vector_add.cuh"

// Создаем параметризованный тестовый класс.
// Параметром будет размер векторов (std::size_t).
class VectorAddComparisonTest : public ::testing::TestWithParam<std::size_t> {
protected:
    // Инициализация генератора случайных чисел один раз для всех тестов
    static void SetUpTestSuite() {
        srand(static_cast<unsigned int>(time(nullptr)));
    }
    
    // Вспомогательная функция для генерации вектора со случайными числами
    void generate_random_vector(std::vector<float>& vec) {
        for (float& val : vec) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
};

// Определяем сам параметризованный тест
TEST_P(VectorAddComparisonTest, GpuVsCpu) {
    // 1. ПОДГОТОВКА (Arrange)
    // Получаем размер вектора из параметров теста
    std::size_t n = GetParam();

    if (n == 0) {
        // Проверяем нулевой размер как граничный случай
        std::vector<float> a, b, c_gpu;
        // Ожидаем, что вызов не вызовет падения
        ASSERT_NO_THROW(vector_add(a.data(), b.data(), c_gpu.data(), n));
        return; // Завершаем тест для n=0
    }
    
    // Создаем векторы на хосте (CPU)
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    
    // Заполняем их случайными данными
    generate_random_vector(h_a);
    generate_random_vector(h_b);

    // Векторы для хранения результатов
    std::vector<float> c_result_gpu(n); // Сюда запишется результат с GPU
    std::vector<float> c_result_cpu(n); // Сюда запишется эталонный результат с CPU

    // 2. ДЕЙСТВИЕ (Act)
    // Выполняем сложение на GPU
    vector_add(h_a.data(), h_b.data(), c_result_gpu.data(), n);
    
    // Выполняем сложение на CPU для получения эталона
    vector_add_cpu(h_a.data(), h_b.data(), c_result_cpu.data(), n);

    // 3. ПРОВЕРКА (Assert)
    // Сравниваем результаты поэлементно
    for (std::size_t i = 0; i < n; ++i) {
        // Используем EXPECT_FLOAT_EQ для корректного сравнения чисел с плавающей точкой
        EXPECT_FLOAT_EQ(c_result_gpu[i], c_result_cpu[i]) 
            << "Mismatch at index " << i; // Сообщение в случае ошибки
    }
}

// "Оживляем" наш тест, передавая ему набор размеров, на которых нужно запуститься.
INSTANTIATE_TEST_SUITE_P(
    VectorAddTestSuite,
    VectorAddComparisonTest,
    ::testing::Values(
        0,          // Граничный случай: нулевой размер
        1,          // Один элемент
        127,        // Чуть меньше размера одного блока (256)
        256,        // Ровно один блок
        257,        // Чуть больше одного блока
        1024,       // Несколько полных блоков
        2000,       // Размер, не кратный размеру блока
        65536       // Большой размер для проверки производительности и стабильности
    )
);