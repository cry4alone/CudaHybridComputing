#include "gtest/gtest.h"
#include "matrix.cuh" 
#include "data.cuh"
#include <Eigen/Dense>
#include <random>
#include <tuple>
#include <vector>

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

class MatMulTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(MatMulTest, Correctness) {
    const auto [m, k, n] = GetParam();

    // Подготовка данных на хосте
    auto host_A_data = generate_random_data(m * k);
    auto host_B_data = generate_random_data(k * n);

    // вычисление с использованием Eigen
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_A(host_A_data.data(), m, k);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_B(host_B_data.data(), k, n);
    Eigen::MatrixXf eigen_C = eigen_A * eigen_B;

    // Вычисление с использованием CUDA
    Matrix A(host_A_data, m, k);
    Matrix B(host_B_data, k, n);
    Matrix C = A * B;
    auto host_C_data = C.to_host();

    // Сравнение результатов
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> my_C_eigen(host_C_data.data(), m, n);

    // isApprox для сравнения с заданной точностью
    const float absolute_tolerance = 1e-5f;
    ASSERT_TRUE(eigen_C.isApprox(my_C_eigen, absolute_tolerance));
}

INSTANTIATE_TEST_SUITE_P(
    MatMulDimensions,
    MatMulTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 3, 127, 128, 129, 512), // m
        ::testing::Values(1, 2, 3, 127, 128, 129, 512), // k
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)  // n
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}