#include "gtest/gtest.h"
#include "data.cuh"
#include "kernels.cuh"
#include <Eigen/Dense>
#include <random>
#include <vector>

static std::vector<float> make_random(size_t n) {
    std::vector<float> v(n);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) v[i] = dis(gen);
    return v;
}

static float eigen_sum(const std::vector<float>& v) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> m(v.data(), v.size());
    return m.sum();
}

class VecRedTest : public ::testing::TestWithParam<size_t> {};

TEST_P(VecRedTest, NoBroadcast) {
    size_t n = GetParam();
    auto h = make_random(n);
    Data d(n, 1);
    d.fill(h);
    VectorView vv(d.device_data(), n);
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));
    launch_vecred_nobr(vv, d_out);
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_NEAR(h_out, eigen_sum(h), 1e-4f);
}

TEST_P(VecRedTest, Broadcast) {
    size_t n = GetParam();
    auto h = make_random(n);
    Data d(n, 1);
    d.fill(h);
    VectorView vv(d.device_data(), n);
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));
    launch_vecred_br(vv, d_out);
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_NEAR(h_out, eigen_sum(h), 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    VecRedTest,
    ::testing::Values(1, 2, 3, 127, 129, 512, 541, 1037)
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}