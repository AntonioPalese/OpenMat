#include "test_helpers.h"

TEST(TensorArithmetic, CPUOperations) {
    Device cpu("cpu:0");
    Tensor<float> a({2}, cpu);
    Tensor<float> b({2}, cpu);

    a({0}) = 1.0f; a({1}) = 2.0f;
    b({0}) = 3.0f; b({1}) = 4.0f;

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

    std::vector<float> host(2);

    (a + b).copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 3.0f); EXPECT_FLOAT_EQ(host[1], 3.0f);

    (a - b).copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], -1.0f); EXPECT_FLOAT_EQ(host[1], -1.0f);

    (a * b).copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 2.0f); EXPECT_FLOAT_EQ(host[1], 2.0f);

    (a / b).copyToHost(host.data());
    EXPECT_FLOAT_EQ(host[0], 0.5f); EXPECT_FLOAT_EQ(host[1], 0.5f);
}

// A(2x3) * B(3x2) = C(2x2)
// A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]  C = [[58,64],[139,154]]
TEST(TensorArithmetic, CPUMatMul) {
    Device cpu("cpu:0");
    Tensor<float> a({2, 3}, cpu);
    a({0,0})=1; a({0,1})=2; a({0,2})=3;
    a({1,0})=4; a({1,1})=5; a({1,2})=6;

    Tensor<float> b({3, 2}, cpu);
    b({0,0})=7;  b({0,1})=8;
    b({1,0})=9;  b({1,1})=10;
    b({2,0})=11; b({2,1})=12;

    Tensor<float> c = a.matmul(b);
    EXPECT_EQ(c.shape()[0], 2u); EXPECT_EQ(c.shape()[1], 2u);
    EXPECT_FLOAT_EQ(c({0,0}), 58.0f);
    EXPECT_FLOAT_EQ(c({0,1}), 64.0f);
    EXPECT_FLOAT_EQ(c({1,0}), 139.0f);
    EXPECT_FLOAT_EQ(c({1,1}), 154.0f);
}

TEST(TensorArithmetic, GPUMatMul) {
    Device gpu("cuda:0");
    Tensor<float> a({2, 2}, gpu);
    Tensor<float> b({2, 2}, gpu);
    a.fill(1.0f);
    b.fill(1.0f);

    std::vector<float> host(4);
    a.matmul(b).copyToHost(host.data());
    for (float v : host) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(TensorArithmetic, GPUFP16Operations) {
    Device gpu("cuda:0");
    Tensor<float16_t> a({4}, gpu);
    Tensor<float16_t> b({4}, gpu);
    a.fill(float16_t(2.0f));
    b.fill(float16_t(3.0f));

    std::vector<float16_t> host(4);

    (a + b).copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), 5.0f, 0.01f);

    (a - b).copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), -1.0f, 0.01f);

    (a * b).copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), 6.0f, 0.01f);

    (a / b).copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), 2.0f/3.0f, 0.01f);
}

TEST(TensorArithmetic, GPUFP16MatMul) {
    Device gpu("cuda:0");
    Tensor<float16_t> a({2, 2}, gpu);
    Tensor<float16_t> b({2, 2}, gpu);
    a.fill(float16_t(1.0f));
    b.fill(float16_t(1.0f));

    Tensor<float16_t> c = a.matmul(b);
    EXPECT_EQ(c.shape()[0], 2u); EXPECT_EQ(c.shape()[1], 2u);

    std::vector<float16_t> host(4);
    c.copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), 2.0f, 0.01f);
}

TEST(TensorArithmetic, GPUFP16MatMulLarger) {
    Device gpu("cuda:0");
    Tensor<float16_t> a({4, 3}, gpu);
    Tensor<float16_t> b({3, 4}, gpu);
    a.fill(float16_t(2.0f));
    b.fill(float16_t(0.5f));

    Tensor<float16_t> c = a.matmul(b);
    EXPECT_EQ(c.shape()[0], 4u); EXPECT_EQ(c.shape()[1], 4u);

    std::vector<float16_t> host(16);
    c.copyToHost(host.data());
    for (auto v : host) EXPECT_NEAR(float(v), 3.0f, 0.01f);
}
