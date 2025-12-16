#pragma once

#include <cuda_runtime.h>

// Host-side reduction of partial sums (device array -> single float)
// Uses Kahan compensated summation in double for numerical stability.
float reduce_on_host(const float* d_partial, int count);
