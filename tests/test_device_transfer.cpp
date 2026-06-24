#include "test_helpers.h"

TEST(DeviceTransfer, CPUtoGPU_Values) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0})=1; a({1})=2; a({2})=3; a({3})=4;

    auto g = a.cuda();
    EXPECT_EQ(g.device_type(), DEVICE_TYPE::CUDA);
    EXPECT_EQ(g.shape(), a.shape());

    auto h = to_host(g);
    EXPECT_FLOAT_EQ(h[0], 1.0f); EXPECT_FLOAT_EQ(h[1], 2.0f);
    EXPECT_FLOAT_EQ(h[2], 3.0f); EXPECT_FLOAT_EQ(h[3], 4.0f);
}

TEST(DeviceTransfer, GPUtoCPU_Values) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    a.fill(7.0f);

    auto c = a.cpu();
    EXPECT_EQ(c.device_type(), DEVICE_TYPE::CPU);
    EXPECT_EQ(c.shape(), a.shape());
    EXPECT_FLOAT_EQ(c({0}), 7.0f);
    EXPECT_FLOAT_EQ(c({3}), 7.0f);
}

TEST(DeviceTransfer, RoundTrip_CPU_GPU_CPU) {
    Device cpu("cpu:0");
    Tensor<float> a({8}, cpu);
    for (size_t i = 0; i < 8; ++i) a({i}) = static_cast<float>(i);

    auto rt = a.cuda().cpu();
    EXPECT_EQ(rt.device_type(), DEVICE_TYPE::CPU);
    for (size_t i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(rt({i}), static_cast<float>(i));
}

TEST(DeviceTransfer, SameDevice_CPU_DeepCopy) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0})=5; a({1})=6; a({2})=7; a({3})=8;

    auto b = a.to(Device("cpu:0"));
    EXPECT_EQ(b.device_type(), DEVICE_TYPE::CPU);
    a({0}) = 99.0f;
    EXPECT_FLOAT_EQ(b({0}), 5.0f);
}

TEST(DeviceTransfer, SameDevice_GPU_DeepCopy) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    a.fill(3.0f);

    auto b = a.to(Device("cuda:0"));
    EXPECT_EQ(b.device_type(), DEVICE_TYPE::CUDA);
    for (float v : to_host(b)) EXPECT_FLOAT_EQ(v, 3.0f);
}

TEST(DeviceTransfer, ShapePreserved_Rank2) {
    Device cpu("cpu:0");
    Tensor<float> a({3, 5}, cpu);
    a.fill(1.0f);

    auto g = a.cuda();
    EXPECT_EQ(g.rank(), 2u);
    EXPECT_EQ(g.shape()[0], 3u); EXPECT_EQ(g.shape()[1], 5u);

    auto back = g.cpu();
    EXPECT_EQ(back.rank(), 2u);
    EXPECT_EQ(back.shape()[0], 3u); EXPECT_EQ(back.shape()[1], 5u);
    EXPECT_FLOAT_EQ(back({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(back({2, 4}), 1.0f);
}

TEST(DeviceTransfer, OpAfterTransfer) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    for (size_t i = 0; i < 4; ++i) a({i}) = static_cast<float>(i + 1);

    auto result = (a.cuda() * 2.0f).cpu();
    EXPECT_FLOAT_EQ(result({0}), 2.0f); EXPECT_FLOAT_EQ(result({1}), 4.0f);
    EXPECT_FLOAT_EQ(result({2}), 6.0f); EXPECT_FLOAT_EQ(result({3}), 8.0f);
}
