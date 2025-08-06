#include <gtest/gtest.h>
#include "tensor.cuh"
#include "mat_utils.h"
#include <vector>

using namespace om;

TEST(TensorArithmetic, CPUOperations) {
    Device cpu("cpu:0");
    Tensor<float> a({2}, cpu);
    Tensor<float> b({2}, cpu);

    a({0}) = 1.0f;
    a({1}) = 2.0f;
    b({0}) = 3.0f;
    b({1}) = 4.0f;

    Tensor<float> add_res = a + b;
    EXPECT_FLOAT_EQ(add_res({0}), 4.0f);
    EXPECT_FLOAT_EQ(add_res({1}), 6.0f);

    Tensor<float> sub_res = a - b;
    EXPECT_FLOAT_EQ(sub_res({0}), -2.0f);
    EXPECT_FLOAT_EQ(sub_res({1}), -2.0f);

    Tensor<float> mul_res = a * b;
    EXPECT_FLOAT_EQ(mul_res({0}), 3.0f);
    EXPECT_FLOAT_EQ(mul_res({1}), 8.0f);

    Tensor<float> div_res = a / b;
    EXPECT_FLOAT_EQ(div_res({0}), 1.0f/3.0f);
    EXPECT_FLOAT_EQ(div_res({1}), 0.5f);
}

TEST(TensorArithmetic, GPUOperations) {
    Device gpu("cuda:0");
    Tensor<float> a({2}, gpu);
    Tensor<float> b({2}, gpu);

    a.fill(1.0f);
    b.fill(2.0f);

    Tensor<float> add_res = a + b;
    Tensor<float> sub_res = a - b;
    Tensor<float> mul_res = a * b;
    Tensor<float> div_res = a / b;

    std::vector<float> host(2);

    add_res.copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 3.0f);
    EXPECT_FLOAT_EQ(host[1], 3.0f);

    sub_res.copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], -1.0f);
    EXPECT_FLOAT_EQ(host[1], -1.0f);

    mul_res.copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 2.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);

    div_res.copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 0.5f);
    EXPECT_FLOAT_EQ(host[1], 0.5f);
}

