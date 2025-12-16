#include <cuda_runtime.h>
#include <vector>
#include "../../includes/vecred_utils.cuh"

float reduce_on_host(const float* d_partial, int count) {
    std::vector<float> h(count);
    cudaMemcpy(h.data(), d_partial, sizeof(float) * count, cudaMemcpyDeviceToHost);
    // Kahan compensated summation in double
    double sum = 0.0;
    double c = 0.0;
    for (int i = 0; i < count; ++i) {
        double y = static_cast<double>(h[i]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return static_cast<float>(sum);
}
