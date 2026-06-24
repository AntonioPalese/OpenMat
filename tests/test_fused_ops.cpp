#include "test_helpers.h"

// ============================================================================
// CPU path
// ============================================================================

TEST(FusedOps, CPUApplyUnary) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0})=1; a({1})=2; a({2})=3; a({3})=4;

    auto r = a.apply(Add<float>{10.0f});
    EXPECT_FLOAT_EQ(r({0}), 11.0f); EXPECT_FLOAT_EQ(r({1}), 12.0f);
    EXPECT_FLOAT_EQ(r({2}), 13.0f); EXPECT_FLOAT_EQ(r({3}), 14.0f);

    auto s = a.apply(Mul<float>{2.0f});
    EXPECT_FLOAT_EQ(s({0}), 2.0f); EXPECT_FLOAT_EQ(s({1}), 4.0f);
    EXPECT_FLOAT_EQ(s({2}), 6.0f); EXPECT_FLOAT_EQ(s({3}), 8.0f);
}

TEST(FusedOps, CPUScaleShift) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0})=1; a({1})=2; a({2})=3; a({3})=4;

    auto r = a.scale_shift(2.0f, 1.0f);  // x*2+1
    EXPECT_FLOAT_EQ(r({0}), 3.0f); EXPECT_FLOAT_EQ(r({1}), 5.0f);
    EXPECT_FLOAT_EQ(r({2}), 7.0f); EXPECT_FLOAT_EQ(r({3}), 9.0f);

    auto s = a.shift_scale(1.0f, 2.0f);  // (x+1)*2
    EXPECT_FLOAT_EQ(s({0}), 4.0f); EXPECT_FLOAT_EQ(s({1}), 6.0f);
    EXPECT_FLOAT_EQ(s({2}), 8.0f); EXPECT_FLOAT_EQ(s({3}), 10.0f);
}

TEST(FusedOps, CPUApplyBinary) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    Tensor<float> b({4}, cpu);
    a({0})=1; a({1})=2; a({2})=3; a({3})=4;
    b({0})=4; b({1})=3; b({2})=2; b({3})=1;

    auto r = a.fused_add_mul(b, 2.0f);
    for (size_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(r({i}), 10.0f);

    auto s = a.fused_mul_add(b, 1.0f);
    EXPECT_FLOAT_EQ(s({0}), 5.0f); EXPECT_FLOAT_EQ(s({1}), 7.0f);
    EXPECT_FLOAT_EQ(s({2}), 7.0f); EXPECT_FLOAT_EQ(s({3}), 5.0f);
}

TEST(FusedOps, CPUScaleShiftKnownValues) {
    Device cpu("cpu:0");
    Tensor<float> a({8}, cpu);
    for (size_t i = 0; i < 8; ++i) a({i}) = static_cast<float>(i + 1);

    auto r = a.scale_shift(3.0f, -1.0f);  // x*3-1
    EXPECT_FLOAT_EQ(r({0}), 2.0f);
    EXPECT_FLOAT_EQ(r({1}), 5.0f);
    EXPECT_FLOAT_EQ(r({2}), 8.0f);
    EXPECT_FLOAT_EQ(r({7}), 23.0f);
}

// ============================================================================
// GPU path — apply (unary)
// ============================================================================

TEST(FusedOps, GPUApplyAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    a.fill(3.0f);
    auto h = to_host(a.apply(Add<float>{7.0f}));
    for (float v : h) EXPECT_FLOAT_EQ(v, 10.0f);
}

TEST(FusedOps, GPUApplyMul_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    a.fill(4.0f);
    auto h = to_host(a.apply(Mul<float>{0.5f}));
    for (float v : h) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(FusedOps, GPUApplyCompose_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    a.fill(1.0f);
    // (x + 2) * 3
    auto h = to_host(a.apply(Compose<Add<float>, Mul<float>>{Add<float>{2.0f}, Mul<float>{3.0f}}));
    for (float v : h) EXPECT_FLOAT_EQ(v, 9.0f);
}

TEST(FusedOps, GPUApplyAdd_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({3, 4}, gpu);
    a.fill(5.0f);
    auto r = a.apply(Add<float>{-2.0f});
    EXPECT_EQ(r.shape()[0], 3u); EXPECT_EQ(r.shape()[1], 4u);
    for (float v : to_host(r)) EXPECT_FLOAT_EQ(v, 3.0f);
}

TEST(FusedOps, GPUApplyMul_Rank3) {
    Device gpu("cuda:0");
    Tensor<float> a({2, 3, 4}, gpu);
    a.fill(2.0f);
    auto r = a.apply(Mul<float>{3.0f});
    EXPECT_EQ(r.rank(), 3u);
    for (float v : to_host(r)) EXPECT_FLOAT_EQ(v, 6.0f);
}

// ============================================================================
// GPU path — scale_shift / shift_scale
// ============================================================================

TEST(FusedOps, GPUScaleShift_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    a.fill(2.0f);
    for (float v : to_host(a.scale_shift(3.0f, 1.0f))) EXPECT_FLOAT_EQ(v, 7.0f);
}

TEST(FusedOps, GPUShiftScale_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    a.fill(2.0f);
    for (float v : to_host(a.shift_scale(1.0f, 3.0f))) EXPECT_FLOAT_EQ(v, 9.0f);
}

TEST(FusedOps, GPUScaleShift_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({4, 4}, gpu);
    a.fill(5.0f);
    for (float v : to_host(a.scale_shift(2.0f, -3.0f))) EXPECT_FLOAT_EQ(v, 7.0f);
}

// ============================================================================
// GPU path — apply_binary (generic)
// ============================================================================

TEST(FusedOps, GPUApplyBinaryAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu); a.fill(3.0f);
    Tensor<float> b({4}, gpu); b.fill(2.0f);
    for (float v : to_host(a.apply_binary(b, BinaryAdd<float>{}))) EXPECT_FLOAT_EQ(v, 5.0f);
}

TEST(FusedOps, GPUApplyBinaryMul_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({3, 3}, gpu); a.fill(4.0f);
    Tensor<float> b({3, 3}, gpu); b.fill(0.5f);
    auto r = a.apply_binary(b, BinaryMul<float>{});
    EXPECT_EQ(r.shape()[0], 3u); EXPECT_EQ(r.shape()[1], 3u);
    for (float v : to_host(r)) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(FusedOps, GPUApplyBinaryDiv_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu); a.fill(9.0f);
    Tensor<float> b({4}, gpu); b.fill(3.0f);
    for (float v : to_host(a.apply_binary(b, BinaryDiv<float>{}))) EXPECT_FLOAT_EQ(v, 3.0f);
}

// ============================================================================
// GPU path — named binary fused methods
// ============================================================================

TEST(FusedOps, GPUFusedAddMul_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu); a.fill(3.0f);
    Tensor<float> b({6}, gpu); b.fill(7.0f);
    for (float v : to_host(a.fused_add_mul(b, 2.0f))) EXPECT_FLOAT_EQ(v, 20.0f);
}

TEST(FusedOps, GPUFusedSubMul_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu); a.fill(10.0f);
    Tensor<float> b({6}, gpu); b.fill(4.0f);
    for (float v : to_host(a.fused_sub_mul(b, 3.0f))) EXPECT_FLOAT_EQ(v, 18.0f);
}

TEST(FusedOps, GPUFusedMulAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu); a.fill(3.0f);
    Tensor<float> b({6}, gpu); b.fill(4.0f);
    for (float v : to_host(a.fused_mul_add(b, 5.0f))) EXPECT_FLOAT_EQ(v, 17.0f);
}

TEST(FusedOps, GPUFusedDivAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu); a.fill(12.0f);
    Tensor<float> b({6}, gpu); b.fill(4.0f);
    for (float v : to_host(a.fused_div_add(b, 1.0f))) EXPECT_FLOAT_EQ(v, 4.0f);
}

TEST(FusedOps, GPUFusedAddMul_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({4, 4}, gpu); a.fill(2.0f);
    Tensor<float> b({4, 4}, gpu); b.fill(3.0f);
    auto r = a.fused_add_mul(b, 0.5f);
    EXPECT_EQ(r.shape()[0], 4u); EXPECT_EQ(r.shape()[1], 4u);
    for (float v : to_host(r)) EXPECT_FLOAT_EQ(v, 2.5f);
}

TEST(FusedOps, GPUFusedMulAdd_Rank3) {
    Device gpu("cuda:0");
    Tensor<float> a({2, 3, 4}, gpu); a.fill(2.0f);
    Tensor<float> b({2, 3, 4}, gpu); b.fill(3.0f);
    auto r = a.fused_mul_add(b, 10.0f);
    EXPECT_EQ(r.rank(), 3u);
    for (float v : to_host(r)) EXPECT_FLOAT_EQ(v, 16.0f);
}

// ============================================================================
// CPU / GPU consistency
// ============================================================================

TEST(FusedOps, CPUGPUConsistency_ScaleShift) {
    const size_t N = 16;
    Device cpu("cpu:0");
    Device gpu("cuda:0");
    Tensor<float> a_cpu({N}, cpu); for (size_t i=0;i<N;++i) a_cpu({i})=5.0f;
    Tensor<float> a_gpu({N}, gpu); a_gpu.fill(5.0f);

    auto r_cpu = a_cpu.scale_shift(2.0f, 3.0f);
    auto h_gpu = to_host(a_gpu.scale_shift(2.0f, 3.0f));

    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(r_cpu({i}), h_gpu[i]);
}

TEST(FusedOps, CPUGPUConsistency_FusedAddMul) {
    const size_t N = 8;
    Device cpu("cpu:0");
    Device gpu("cuda:0");
    Tensor<float> a_cpu({N}, cpu); for (size_t i=0;i<N;++i) a_cpu({i})=4.0f;
    Tensor<float> b_cpu({N}, cpu); for (size_t i=0;i<N;++i) b_cpu({i})=6.0f;
    Tensor<float> a_gpu({N}, gpu); a_gpu.fill(4.0f);
    Tensor<float> b_gpu({N}, gpu); b_gpu.fill(6.0f);

    auto r_cpu = a_cpu.fused_add_mul(b_cpu, 0.5f);
    auto h_gpu = to_host(a_gpu.fused_add_mul(b_gpu, 0.5f));

    for (size_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(r_cpu({i}), h_gpu[i]);
}

// ============================================================================
// Equivalenza con operazioni separate
// ============================================================================

TEST(FusedOps, FusedAddMulEquivalent) {
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu); a.fill(3.0f);
    Tensor<float> b({8}, gpu); b.fill(5.0f);
    auto h_f = to_host(a.fused_add_mul(b, 2.0f));
    auto h_u = to_host((a + b) * 2.0f);
    for (size_t i = 0; i < h_f.size(); ++i) EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

TEST(FusedOps, FusedMulAddEquivalent) {
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu); a.fill(3.0f);
    Tensor<float> b({8}, gpu); b.fill(4.0f);
    auto h_f = to_host(a.fused_mul_add(b, 2.0f));
    auto h_u = to_host((a * b) + 2.0f);
    for (size_t i = 0; i < h_f.size(); ++i) EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

TEST(FusedOps, ScaleShiftEquivalent) {
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu); a.fill(5.0f);
    auto h_f = to_host(a.scale_shift(2.0f, -1.0f));
    auto h_u = to_host(a * 2.0f + (-1.0f));
    for (size_t i = 0; i < h_f.size(); ++i) EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

// ============================================================================
// Errori attesi
// ============================================================================

TEST(FusedOps, ShapeMismatchThrows) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu); a.fill(1.0f);
    Tensor<float> b({8}, gpu); b.fill(1.0f);
    EXPECT_THROW(a.apply_binary(b, BinaryAdd<float>{}), std::invalid_argument);
}
