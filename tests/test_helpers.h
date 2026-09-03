#pragma once
#include <gtest/gtest.h>
#include "tensor.cuh"
#include "mat_utils.h"
#include <vector>

using namespace om;

inline std::vector<float> to_host(const Tensor<float>& t) {
    std::vector<float> v(t.size());
    t.copyToHost(v.data());
    return v;
}

// Skips the running test when no CUDA device is usable. Every test that touches
// the GPU starts with this, so the full suite is runnable on a machine without a
// device (and in the CPU-only CI job) instead of reporting a wall of failures.
// Placed as the first statement of the test body; GTEST_SKIP() returns.
#define OM_REQUIRE_CUDA()                                            \
    do {                                                             \
        int _om_device_count = 0;                                    \
        if (cudaGetDeviceCount(&_om_device_count) != cudaSuccess ||  \
            _om_device_count == 0) {                                 \
            cudaGetLastError();                                      \
            GTEST_SKIP() << "no CUDA device available";              \
        }                                                            \
    } while (0)
