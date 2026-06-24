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
