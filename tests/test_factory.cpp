#include "test_helpers.h"

// ============================================================================
// zeros
// ============================================================================

TEST(TensorFactory, ZerosCPU) {
    auto t = Tensor<float>::zeros({3, 4});
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CPU);
    EXPECT_EQ(t.rank(), 2u);
    EXPECT_EQ(t.shape()[0], 3u); EXPECT_EQ(t.shape()[1], 4u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(t({i, j}), 0.0f);
}

TEST(TensorFactory, ZerosGPU) {
    auto t = Tensor<float>::zeros({8}, Device("cuda:0"));
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CUDA);
    for (float v : to_host(t)) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(TensorFactory, ZerosInt) {
    auto t = Tensor<int>::zeros({5}, Device("cpu:0"));
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(t({i}), 0);
}

// ============================================================================
// ones
// ============================================================================

TEST(TensorFactory, OnesCPU) {
    auto t = Tensor<float>::ones({2, 3});
    EXPECT_EQ(t.rank(), 2u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t({i, j}), 1.0f);
}

TEST(TensorFactory, OnesGPU) {
    auto t = Tensor<float>::ones({8}, Device("cuda:0"));
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CUDA);
    for (float v : to_host(t)) EXPECT_FLOAT_EQ(v, 1.0f);
}

// ============================================================================
// full
// ============================================================================

TEST(TensorFactory, FullCPU) {
    auto t = Tensor<float>::full({4}, 3.14f);
    for (size_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(t({i}), 3.14f);
}

TEST(TensorFactory, FullGPU) {
    auto t = Tensor<float>::full({6}, -2.0f, Device("cuda:0"));
    for (float v : to_host(t)) EXPECT_FLOAT_EQ(v, -2.0f);
}

TEST(TensorFactory, FullNegativeValue) {
    auto t = Tensor<float>::full({3, 3}, -7.5f, Device("cpu:0"));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t({i, j}), -7.5f);
}

// ============================================================================
// from_vector
// ============================================================================

TEST(TensorFactory, FromVectorCPU_Rank1) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor<float>::from_vector(data, {4});
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CPU);
    EXPECT_EQ(t.size(), 4u);
    for (size_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(t({i}), data[i]);
}

TEST(TensorFactory, FromVectorCPU_Rank2) {
    // row-major: [[1,2,3],[4,5,6]]
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto t = Tensor<float>::from_vector(data, {2, 3});
    EXPECT_EQ(t.shape()[0], 2u); EXPECT_EQ(t.shape()[1], 3u);
    EXPECT_FLOAT_EQ(t({0, 0}), 1.0f); EXPECT_FLOAT_EQ(t({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t({1, 0}), 4.0f); EXPECT_FLOAT_EQ(t({1, 2}), 6.0f);
}

TEST(TensorFactory, FromVectorGPU) {
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f};
    auto t = Tensor<float>::from_vector(data, {4}, Device("cuda:0"));
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CUDA);
    auto h = to_host(t);
    for (size_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(h[i], data[i]);
}

TEST(TensorFactory, FromVectorInt) {
    std::vector<int> data = {5, 10, 15};
    auto t = Tensor<int>::from_vector(data, {3});
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(t({i}), data[i]);
}

TEST(TensorFactory, FromVectorSizeMismatchThrows) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    EXPECT_THROW(Tensor<float>::from_vector(data, {4}), std::invalid_argument);
}

// ============================================================================
// from_vector + operazioni: integrazione con il resto dell'API
// ============================================================================

TEST(TensorFactory, FromVectorThenOp_CPU) {
    auto a = Tensor<float>::from_vector({1, 2, 3, 4}, {4});
    auto b = Tensor<float>::ones({4});
    auto c = a + b;
    EXPECT_FLOAT_EQ(c({0}), 2.0f); EXPECT_FLOAT_EQ(c({3}), 5.0f);
}

TEST(TensorFactory, FromVectorThenOp_GPU) {
    auto a = Tensor<float>::from_vector({1, 2, 3, 4}, {4}, Device("cuda:0"));
    auto b = Tensor<float>::ones({4}, Device("cuda:0"));
    auto h = to_host(a + b);
    EXPECT_FLOAT_EQ(h[0], 2.0f); EXPECT_FLOAT_EQ(h[3], 5.0f);
}

TEST(TensorFactory, FromVectorMatmul) {
    // A = [[1,0],[0,1]] (identità), B = [[5,6],[7,8]] → C = B
    auto A = Tensor<float>::from_vector({1,0,0,1}, {2,2}, Device("cuda:0"));
    auto B = Tensor<float>::from_vector({5,6,7,8}, {2,2}, Device("cuda:0"));
    auto h = to_host(A.matmul(B));
    EXPECT_FLOAT_EQ(h[0], 5.0f); EXPECT_FLOAT_EQ(h[1], 6.0f);
    EXPECT_FLOAT_EQ(h[2], 7.0f); EXPECT_FLOAT_EQ(h[3], 8.0f);
}
