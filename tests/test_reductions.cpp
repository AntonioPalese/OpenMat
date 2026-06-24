#include "test_helpers.h"

TEST(Reductions, SumCPU_Rank1) {
    auto t = Tensor<float>::from_vector({1, 2, 3, 4, 5}, {5});
    EXPECT_FLOAT_EQ(t.sum(), 15.0f);
}

TEST(Reductions, SumCPU_Rank2) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    EXPECT_FLOAT_EQ(t.sum(), 21.0f);
}

TEST(Reductions, SumGPU_Rank1) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5}, {5}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.sum(), 15.0f);
}

TEST(Reductions, SumGPU_Uniform) {
    auto t = Tensor<float>::full({1024}, 2.0f, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.sum(), 2048.0f);
}

TEST(Reductions, SumGPU_LargeArray) {
    const size_t N = 1 << 20;
    auto t = Tensor<float>::ones({N}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.sum(), static_cast<float>(N));
}

TEST(Reductions, SumCPUMatchesGPU) {
    std::vector<float> data(256);
    for (size_t i = 0; i < 256; ++i) data[i] = static_cast<float>(i);
    auto cpu = Tensor<float>::from_vector(data, {256});
    auto gpu = Tensor<float>::from_vector(data, {256}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(cpu.sum(), gpu.sum());
}

TEST(Reductions, MeanCPU) {
    auto t = Tensor<float>::from_vector({2, 4, 6, 8}, {4});
    EXPECT_FLOAT_EQ(t.mean(), 5.0f);
}

TEST(Reductions, MeanGPU) {
    auto t = Tensor<float>::full({100}, 3.0f, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.mean(), 3.0f);
}

TEST(Reductions, MeanCPUMatchesGPU) {
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i) data[i] = static_cast<float>(i + 1);
    auto cpu = Tensor<float>::from_vector(data, {64});
    auto gpu = Tensor<float>::from_vector(data, {64}, Device("cuda:0"));
    EXPECT_NEAR(cpu.mean(), gpu.mean(), 1e-4f);
}

TEST(Reductions, MinCPU) {
    auto t = Tensor<float>::from_vector({5, 1, 3, 2, 4}, {5});
    EXPECT_FLOAT_EQ(t.min(), 1.0f);
}

TEST(Reductions, MinGPU) {
    auto t = Tensor<float>::from_vector({5,1,3,2,4}, {5}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.min(), 1.0f);
}

TEST(Reductions, MinGPU_NegativeValues) {
    auto t = Tensor<float>::from_vector({0.0f, -3.0f, 2.0f, -1.0f}, {4}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.min(), -3.0f);
}

TEST(Reductions, MinGPU_LargeArray) {
    const size_t N = 1 << 20;
    auto t = Tensor<float>::full({N}, 7.0f, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.min(), 7.0f);
}

TEST(Reductions, MinCPUMatchesGPU) {
    std::vector<float> data(128);
    for (size_t i = 0; i < 128; ++i) data[i] = static_cast<float>(128 - i);
    auto cpu = Tensor<float>::from_vector(data, {128});
    auto gpu = Tensor<float>::from_vector(data, {128}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(cpu.min(), gpu.min());
}

TEST(Reductions, MaxCPU) {
    auto t = Tensor<float>::from_vector({5,1,3,2,4}, {5});
    EXPECT_FLOAT_EQ(t.max(), 5.0f);
}

TEST(Reductions, MaxGPU) {
    auto t = Tensor<float>::from_vector({5,1,3,2,4}, {5}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.max(), 5.0f);
}

TEST(Reductions, MaxGPU_NegativeValues) {
    auto t = Tensor<float>::from_vector({-5.0f, -1.0f, -3.0f}, {3}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.max(), -1.0f);
}

TEST(Reductions, MaxGPU_LargeArray) {
    const size_t N = 1 << 20;
    auto t = Tensor<float>::full({N}, -4.0f, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.max(), -4.0f);
}

TEST(Reductions, MaxCPUMatchesGPU) {
    std::vector<float> data(128);
    for (size_t i = 0; i < 128; ++i) data[i] = static_cast<float>(i);
    auto cpu = Tensor<float>::from_vector(data, {128});
    auto gpu = Tensor<float>::from_vector(data, {128}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(cpu.max(), gpu.max());
}

TEST(Reductions, SumAfterScaleShift) {
    auto t = Tensor<float>::from_vector({1,2,3,4}, {4}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.scale_shift(2.0f, 1.0f).sum(), 24.0f);
}

TEST(Reductions, MinMaxRange) {
    auto t = Tensor<float>::from_vector({3,1,4,1,5,9,2,6}, {8}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.min(), 1.0f);
    EXPECT_FLOAT_EQ(t.max(), 9.0f);
}

TEST(Reductions, MeanOfOnes) {
    auto t = Tensor<float>::ones({1000}, Device("cuda:0"));
    EXPECT_NEAR(t.mean(), 1.0f, 1e-5f);
}

TEST(Reductions, SumInt) {
    auto t = Tensor<int>::from_vector({1,2,3,4,5}, {5});
    EXPECT_EQ(t.sum(), 15);
}
