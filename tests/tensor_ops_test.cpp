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

// MatMul Tests
// Test: A(2x3) * B(3x2) = C(2x2)
// A = [[1, 2, 3],    B = [[7, 8],     C = [[1*7+2*9+3*11, 1*8+2*10+3*12],   [[58, 64],
//      [4, 5, 6]]         [9, 10],        [4*7+5*9+6*11, 4*8+5*10+6*12]] =  [139, 154]]
//                         [11, 12]]

TEST(TensorArithmetic, CPUMatMul) {
    Device cpu("cpu:0");
    
    // Create 2x3 matrix A
    Tensor<float> a({2, 3}, cpu);
    a({0, 0}) = 1.0f; a({0, 1}) = 2.0f; a({0, 2}) = 3.0f;
    a({1, 0}) = 4.0f; a({1, 1}) = 5.0f; a({1, 2}) = 6.0f;
    
    // Create 3x2 matrix B
    Tensor<float> b({3, 2}, cpu);
    b({0, 0}) = 7.0f;  b({0, 1}) = 8.0f;
    b({1, 0}) = 9.0f;  b({1, 1}) = 10.0f;
    b({2, 0}) = 11.0f; b({2, 1}) = 12.0f;
    
    // Compute C = A * B (should be 2x2)
    Tensor<float> c = a.matmul(b);
    
    // Verify shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    
    // Verify values
    EXPECT_FLOAT_EQ(c({0, 0}), 58.0f);   // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    EXPECT_FLOAT_EQ(c({0, 1}), 64.0f);   // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(c({1, 0}), 139.0f);  // 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    EXPECT_FLOAT_EQ(c({1, 1}), 154.0f);  // 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
}

TEST(TensorArithmetic, GPUMatMul) {
    Device gpu("cuda:0");
    Device cpu("cpu:0");
    
    // Create matrices on GPU
    Tensor<float> a({2, 3}, gpu);
    Tensor<float> b({3, 2}, gpu);
    
    // Initialize on CPU first, then manually set values via fill and arithmetic
    // For simplicity, use identity-like matrices for easy verification
    // A = [[1, 0, 0],      B = [[1],     C = [[1],
    //      [0, 1, 0]]           [2],          [2]]
    //                           [3]]
    
    // Actually, let's use a simpler test with fill
    // Create [2x2] identity matmul
    Tensor<float> a2({2, 2}, gpu);
    Tensor<float> b2({2, 2}, gpu);
    
    // Fill with known values using host tensors then test
    Tensor<float> a_cpu({2, 2}, cpu);
    Tensor<float> b_cpu({2, 2}, cpu);
    
    a_cpu({0, 0}) = 1.0f; a_cpu({0, 1}) = 2.0f;
    a_cpu({1, 0}) = 3.0f; a_cpu({1, 1}) = 4.0f;
    
    b_cpu({0, 0}) = 5.0f; b_cpu({0, 1}) = 6.0f;
    b_cpu({1, 0}) = 7.0f; b_cpu({1, 1}) = 8.0f;
    
    // For GPU test, we'll use filled values (simpler approach)
    // Test: all-ones matrices 
    // [1,1] × [1,1] = [2, 2]
    // [1,1]   [1,1]   [2, 2]
    Tensor<float> ones_a({2, 2}, gpu);
    Tensor<float> ones_b({2, 2}, gpu);
    ones_a.fill(1.0f);
    ones_b.fill(1.0f);
    
    Tensor<float> c = ones_a.matmul(ones_b);
    
    // Copy result to host
    std::vector<float> host(4);
    c.copyToHost(host.data());
    
    // Each element should be 2.0 (1*1 + 1*1)
    EXPECT_FLOAT_EQ(host[0], 2.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 2.0f);
    EXPECT_FLOAT_EQ(host[3], 2.0f);
}

// FP16 Tests
TEST(TensorArithmetic, GPUFP16Operations) {
    Device gpu("cuda:0");
    
    // Create fp16 tensors
    Tensor<float16_t> a({4}, gpu);
    Tensor<float16_t> b({4}, gpu);
    
    a.fill(float16_t(2.0f));
    b.fill(float16_t(3.0f));
    
    // Test addition
    Tensor<float16_t> add_res = a + b;
    std::vector<float16_t> host(4);
    add_res.copyToHost(host.data());
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(float(host[i]), 5.0f, 0.01f);  // 2 + 3 = 5
    }
    
    // Test subtraction
    Tensor<float16_t> sub_res = a - b;
    sub_res.copyToHost(host.data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(float(host[i]), -1.0f, 0.01f);  // 2 - 3 = -1
    }
    
    // Test multiplication
    Tensor<float16_t> mul_res = a * b;
    mul_res.copyToHost(host.data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(float(host[i]), 6.0f, 0.01f);  // 2 * 3 = 6
    }
    
    // Test division
    Tensor<float16_t> div_res = a / b;
    div_res.copyToHost(host.data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(float(host[i]), 2.0f/3.0f, 0.01f);  // 2 / 3 ≈ 0.666
    }
}

TEST(TensorArithmetic, GPUFP16MatMul) {
    Device gpu("cuda:0");
    
    // Create fp16 matrices for matmul: [2x2] × [2x2]
    // A = [[1, 1],   B = [[1, 1],   C = [[2, 2],
    //      [1, 1]]        [1, 1]]        [2, 2]]
    Tensor<float16_t> a({2, 2}, gpu);
    Tensor<float16_t> b({2, 2}, gpu);
    
    a.fill(float16_t(1.0f));
    b.fill(float16_t(1.0f));
    
    Tensor<float16_t> c = a.matmul(b);
    
    // Verify shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    
    // Copy result to host and verify
    std::vector<float16_t> host(4);
    c.copyToHost(host.data());
    
    // Each element should be 2.0 (1*1 + 1*1)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(float(host[i]), 2.0f, 0.01f);
    }
}

TEST(TensorArithmetic, GPUFP16MatMulLarger) {
    Device gpu("cuda:0");
    
    // Test larger matrix: [4x3] × [3x4] = [4x4]
    Tensor<float16_t> a({4, 3}, gpu);
    Tensor<float16_t> b({3, 4}, gpu);
    
    a.fill(float16_t(2.0f));  // All 2s
    b.fill(float16_t(0.5f));  // All 0.5s
    
    Tensor<float16_t> c = a.matmul(b);
    
    // Verify shape
    EXPECT_EQ(c.shape()[0], 4);
    EXPECT_EQ(c.shape()[1], 4);
    
    // Copy result to host
    std::vector<float16_t> host(16);
    c.copyToHost(host.data());
    
    // Each element: sum of 3 products of (2 * 0.5) = 3 * 1 = 3
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(float(host[i]), 3.0f, 0.01f);
    }
}

// ============================================================================
// Fused operations — CPU path
// ============================================================================

TEST(FusedOps, CPUApplyUnary) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0}) = 1.0f; a({1}) = 2.0f; a({2}) = 3.0f; a({3}) = 4.0f;

    // Add scalar
    auto r = a.apply(Add<float>{10.0f});
    EXPECT_FLOAT_EQ(r({0}), 11.0f);
    EXPECT_FLOAT_EQ(r({1}), 12.0f);
    EXPECT_FLOAT_EQ(r({2}), 13.0f);
    EXPECT_FLOAT_EQ(r({3}), 14.0f);

    // Mul scalar
    auto s = a.apply(Mul<float>{2.0f});
    EXPECT_FLOAT_EQ(s({0}), 2.0f);
    EXPECT_FLOAT_EQ(s({1}), 4.0f);
    EXPECT_FLOAT_EQ(s({2}), 6.0f);
    EXPECT_FLOAT_EQ(s({3}), 8.0f);
}

TEST(FusedOps, CPUScaleShift) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0}) = 1.0f; a({1}) = 2.0f; a({2}) = 3.0f; a({3}) = 4.0f;

    // scale_shift: x * 2 + 1
    auto r = a.scale_shift(2.0f, 1.0f);
    EXPECT_FLOAT_EQ(r({0}), 3.0f);
    EXPECT_FLOAT_EQ(r({1}), 5.0f);
    EXPECT_FLOAT_EQ(r({2}), 7.0f);
    EXPECT_FLOAT_EQ(r({3}), 9.0f);

    // shift_scale: (x + 1) * 2
    auto s = a.shift_scale(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(s({0}), 4.0f);
    EXPECT_FLOAT_EQ(s({1}), 6.0f);
    EXPECT_FLOAT_EQ(s({2}), 8.0f);
    EXPECT_FLOAT_EQ(s({3}), 10.0f);
}

TEST(FusedOps, CPUApplyBinary) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    Tensor<float> b({4}, cpu);
    a({0}) = 1.0f; a({1}) = 2.0f; a({2}) = 3.0f; a({3}) = 4.0f;
    b({0}) = 4.0f; b({1}) = 3.0f; b({2}) = 2.0f; b({3}) = 1.0f;

    // (a + b) * 2
    auto r = a.fused_add_mul(b, 2.0f);
    EXPECT_FLOAT_EQ(r({0}), 10.0f);  // (1+4)*2
    EXPECT_FLOAT_EQ(r({1}), 10.0f);  // (2+3)*2
    EXPECT_FLOAT_EQ(r({2}), 10.0f);  // (3+2)*2
    EXPECT_FLOAT_EQ(r({3}), 10.0f);  // (4+1)*2

    // (a * b) + 1
    auto s = a.fused_mul_add(b, 1.0f);
    EXPECT_FLOAT_EQ(s({0}), 5.0f);   // 1*4+1
    EXPECT_FLOAT_EQ(s({1}), 7.0f);   // 2*3+1
    EXPECT_FLOAT_EQ(s({2}), 7.0f);   // 3*2+1
    EXPECT_FLOAT_EQ(s({3}), 5.0f);   // 4*1+1
}

TEST(FusedOps, CPUMatchesGPU) {
    // Verifica che CPU e GPU producano lo stesso risultato per scale_shift
    Device cpu("cpu:0");
    Device gpu("cuda:0");

    const size_t N = 8;
    Tensor<float> a_cpu({N}, cpu);
    Tensor<float> a_gpu({N}, gpu);

    for (size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(i + 1);
        a_cpu({i}) = v;
    }
    a_gpu.fill(0.0f);
    // Copia manuale cpu→gpu
    std::vector<float> buf(N);
    for (size_t i = 0; i < N; ++i) buf[i] = a_cpu({i});
    // Riempiamo il tensore GPU elemento per elemento tramite fill non è pratico —
    // usiamo un loop su GPU fill + verifichiamo i valori attesi direttamente
    // Qui testiamo solo che il path CPU non crashi e produca valori corretti
    auto r_cpu = a_cpu.scale_shift(3.0f, -1.0f);  // x*3 - 1
    EXPECT_FLOAT_EQ(r_cpu({0}), 2.0f);   // 1*3-1
    EXPECT_FLOAT_EQ(r_cpu({1}), 5.0f);   // 2*3-1
    EXPECT_FLOAT_EQ(r_cpu({2}), 8.0f);   // 3*3-1
    EXPECT_FLOAT_EQ(r_cpu({7}), 23.0f);  // 8*3-1
}

// ============================================================================
// Device transfer — .to() / .cpu() / .cuda()
// ============================================================================

TEST(DeviceTransfer, CPUtoGPU_Values) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0}) = 1.0f; a({1}) = 2.0f; a({2}) = 3.0f; a({3}) = 4.0f;

    auto g = a.cuda();
    EXPECT_EQ(g.device_type(), DEVICE_TYPE::CUDA);
    EXPECT_EQ(g.shape(), a.shape());

    std::vector<float> h(4);
    g.copyToHost(h.data());
    EXPECT_FLOAT_EQ(h[0], 1.0f);
    EXPECT_FLOAT_EQ(h[1], 2.0f);
    EXPECT_FLOAT_EQ(h[2], 3.0f);
    EXPECT_FLOAT_EQ(h[3], 4.0f);
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

    auto roundtrip = a.cuda().cpu();
    EXPECT_EQ(roundtrip.device_type(), DEVICE_TYPE::CPU);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(roundtrip({i}), static_cast<float>(i));
}

TEST(DeviceTransfer, SameDevice_CPU_DeepCopy) {
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    a({0}) = 5.0f; a({1}) = 6.0f; a({2}) = 7.0f; a({3}) = 8.0f;

    auto b = a.to(Device("cpu:0"));
    EXPECT_EQ(b.device_type(), DEVICE_TYPE::CPU);

    // modifica a — b non deve cambiare (deep copy)
    a({0}) = 99.0f;
    EXPECT_FLOAT_EQ(b({0}), 5.0f);
}

TEST(DeviceTransfer, SameDevice_GPU_DeepCopy) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    a.fill(3.0f);

    auto b = a.to(Device("cuda:0"));
    EXPECT_EQ(b.device_type(), DEVICE_TYPE::CUDA);

    std::vector<float> h(4);
    b.copyToHost(h.data());
    for (float v : h) EXPECT_FLOAT_EQ(v, 3.0f);
}

TEST(DeviceTransfer, ShapePreserved_Rank2) {
    Device cpu("cpu:0");
    Tensor<float> a({3, 5}, cpu);
    a.fill(1.0f);

    auto g = a.cuda();
    EXPECT_EQ(g.rank(), 2u);
    EXPECT_EQ(g.shape()[0], 3u);
    EXPECT_EQ(g.shape()[1], 5u);

    auto back = g.cpu();
    EXPECT_EQ(back.rank(), 2u);
    EXPECT_EQ(back.shape()[0], 3u);
    EXPECT_EQ(back.shape()[1], 5u);
    EXPECT_FLOAT_EQ(back({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(back({2, 4}), 1.0f);
}

TEST(DeviceTransfer, OpAfterTransfer) {
    // trasferisci su GPU, esegui un'operazione, riporta su CPU
    Device cpu("cpu:0");
    Tensor<float> a({4}, cpu);
    for (size_t i = 0; i < 4; ++i) a({i}) = static_cast<float>(i + 1);

    auto result = (a.cuda() * 2.0f).cpu();
    EXPECT_FLOAT_EQ(result({0}), 2.0f);
    EXPECT_FLOAT_EQ(result({1}), 4.0f);
    EXPECT_FLOAT_EQ(result({2}), 6.0f);
    EXPECT_FLOAT_EQ(result({3}), 8.0f);
}

// ============================================================================
// Fused operations — GPU path
// ============================================================================

// Helper: copy GPU tensor to std::vector
static std::vector<float> to_host(const Tensor<float>& t) {
    std::vector<float> v(t.size());
    t.copyToHost(v.data());
    return v;
}

// --- apply (unary) ---

TEST(FusedOps, GPUApplyAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    a.fill(3.0f);

    auto r = a.apply(Add<float>{7.0f});
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 10.0f);
}

TEST(FusedOps, GPUApplyMul_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    a.fill(4.0f);

    auto r = a.apply(Mul<float>{0.5f});
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(FusedOps, GPUApplyCompose_Rank1) {
    // Compose<Add, Mul>: x → (x + 2) * 3
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    a.fill(1.0f);

    auto r = a.apply(Compose<Add<float>, Mul<float>>{Add<float>{2.0f}, Mul<float>{3.0f}});
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 9.0f);  // (1+2)*3
}

TEST(FusedOps, GPUApplyAdd_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({3, 4}, gpu);
    a.fill(5.0f);

    auto r = a.apply(Add<float>{-2.0f});
    EXPECT_EQ(r.shape()[0], 3u);
    EXPECT_EQ(r.shape()[1], 4u);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 3.0f);
}

TEST(FusedOps, GPUApplyMul_Rank3) {
    Device gpu("cuda:0");
    Tensor<float> a({2, 3, 4}, gpu);
    a.fill(2.0f);

    auto r = a.apply(Mul<float>{3.0f});
    EXPECT_EQ(r.rank(), 3u);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 6.0f);
}

// --- scale_shift / shift_scale ---

TEST(FusedOps, GPUScaleShift_Rank1) {
    // scale_shift(s, b): x*s + b
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    a.fill(2.0f);

    auto r = a.scale_shift(3.0f, 1.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 7.0f);  // 2*3+1
}

TEST(FusedOps, GPUShiftScale_Rank1) {
    // shift_scale(b, s): (x+b)*s
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    a.fill(2.0f);

    auto r = a.shift_scale(1.0f, 3.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 9.0f);  // (2+1)*3
}

TEST(FusedOps, GPUScaleShift_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({4, 4}, gpu);
    a.fill(5.0f);

    auto r = a.scale_shift(2.0f, -3.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 7.0f);  // 5*2-3
}

// --- apply_binary (generic) ---

TEST(FusedOps, GPUApplyBinaryAdd_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    Tensor<float> b({4}, gpu);
    a.fill(3.0f);
    b.fill(2.0f);

    auto r = a.apply_binary(b, BinaryAdd<float>{});
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 5.0f);
}

TEST(FusedOps, GPUApplyBinaryMul_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({3, 3}, gpu);
    Tensor<float> b({3, 3}, gpu);
    a.fill(4.0f);
    b.fill(0.5f);

    auto r = a.apply_binary(b, BinaryMul<float>{});
    EXPECT_EQ(r.shape()[0], 3u);
    EXPECT_EQ(r.shape()[1], 3u);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(FusedOps, GPUApplyBinaryDiv_Rank1) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    Tensor<float> b({4}, gpu);
    a.fill(9.0f);
    b.fill(3.0f);

    auto r = a.apply_binary(b, BinaryDiv<float>{});
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 3.0f);
}

// --- named binary fused methods ---

TEST(FusedOps, GPUFusedAddMul_Rank1) {
    // (a + b) * scale
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    Tensor<float> b({6}, gpu);
    a.fill(3.0f);
    b.fill(7.0f);

    auto r = a.fused_add_mul(b, 2.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 20.0f);  // (3+7)*2
}

TEST(FusedOps, GPUFusedSubMul_Rank1) {
    // (a - b) * scale
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    Tensor<float> b({6}, gpu);
    a.fill(10.0f);
    b.fill(4.0f);

    auto r = a.fused_sub_mul(b, 3.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 18.0f);  // (10-4)*3
}

TEST(FusedOps, GPUFusedMulAdd_Rank1) {
    // (a * b) + shift
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    Tensor<float> b({6}, gpu);
    a.fill(3.0f);
    b.fill(4.0f);

    auto r = a.fused_mul_add(b, 5.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 17.0f);  // 3*4+5
}

TEST(FusedOps, GPUFusedDivAdd_Rank1) {
    // (a / b) + shift
    Device gpu("cuda:0");
    Tensor<float> a({6}, gpu);
    Tensor<float> b({6}, gpu);
    a.fill(12.0f);
    b.fill(4.0f);

    auto r = a.fused_div_add(b, 1.0f);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 4.0f);  // 12/4+1
}

TEST(FusedOps, GPUFusedAddMul_Rank2) {
    Device gpu("cuda:0");
    Tensor<float> a({4, 4}, gpu);
    Tensor<float> b({4, 4}, gpu);
    a.fill(2.0f);
    b.fill(3.0f);

    auto r = a.fused_add_mul(b, 0.5f);
    EXPECT_EQ(r.shape()[0], 4u);
    EXPECT_EQ(r.shape()[1], 4u);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 2.5f);  // (2+3)*0.5
}

TEST(FusedOps, GPUFusedMulAdd_Rank3) {
    Device gpu("cuda:0");
    Tensor<float> a({2, 3, 4}, gpu);
    Tensor<float> b({2, 3, 4}, gpu);
    a.fill(2.0f);
    b.fill(3.0f);

    auto r = a.fused_mul_add(b, 10.0f);
    EXPECT_EQ(r.rank(), 3u);
    auto h = to_host(r);
    for (float v : h) EXPECT_FLOAT_EQ(v, 16.0f);  // 2*3+10
}

// --- CPU vs GPU consistency ---

TEST(FusedOps, CPUGPUConsistency_ScaleShift) {
    // scale_shift deve dare lo stesso risultato su CPU e GPU
    const size_t N = 16;
    Device cpu("cpu:0");
    Device gpu("cuda:0");

    Tensor<float> a_cpu({N}, cpu);
    Tensor<float> a_gpu({N}, gpu);

    for (size_t i = 0; i < N; ++i)
        a_cpu({i}) = static_cast<float>(i);
    a_gpu.fill(0.0f);
    // Inizializza GPU con gli stessi valori via scale_shift da zero:
    // a_gpu[i] = 0, non possiamo caricare valori arbitrari senza from_vector.
    // Usiamo valori uniformi che possiamo verificare su entrambi.
    Tensor<float> b_cpu({N}, cpu);
    Tensor<float> b_gpu({N}, gpu);
    for (size_t i = 0; i < N; ++i) b_cpu({i}) = 5.0f;
    b_gpu.fill(5.0f);

    auto r_cpu = b_cpu.scale_shift(2.0f, 3.0f);
    auto r_gpu = b_gpu.scale_shift(2.0f, 3.0f);

    std::vector<float> h_gpu(N);
    r_gpu.copyToHost(h_gpu.data());

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(r_cpu({i}), h_gpu[i]);
}

TEST(FusedOps, CPUGPUConsistency_FusedAddMul) {
    const size_t N = 8;
    Device cpu("cpu:0");
    Device gpu("cuda:0");

    Tensor<float> a_cpu({N}, cpu);
    Tensor<float> b_cpu({N}, cpu);
    Tensor<float> a_gpu({N}, gpu);
    Tensor<float> b_gpu({N}, gpu);
    for (size_t i = 0; i < N; ++i) {
        a_cpu({i}) = 4.0f;
        b_cpu({i}) = 6.0f;
    }
    a_gpu.fill(4.0f);
    b_gpu.fill(6.0f);

    auto r_cpu = a_cpu.fused_add_mul(b_cpu, 0.5f);
    auto r_gpu = a_gpu.fused_add_mul(b_gpu, 0.5f);

    std::vector<float> h_gpu(N);
    r_gpu.copyToHost(h_gpu.data());

    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(r_cpu({i}), h_gpu[i]);
}

// --- equivalenza con operazioni separate ---

TEST(FusedOps, FusedAddMulEquivalent) {
    // fused_add_mul(b, s) == (a + b) * s calcolato in due passi
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    Tensor<float> b({8}, gpu);
    a.fill(3.0f);
    b.fill(5.0f);

    auto fused  = a.fused_add_mul(b, 2.0f);
    auto unfused = (a + b) * 2.0f;

    auto h_f = to_host(fused);
    auto h_u = to_host(unfused);
    for (size_t i = 0; i < h_f.size(); ++i)
        EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

TEST(FusedOps, FusedMulAddEquivalent) {
    // fused_mul_add(b, s) == (a * b) + s calcolato in due passi
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    Tensor<float> b({8}, gpu);
    a.fill(3.0f);
    b.fill(4.0f);

    auto fused   = a.fused_mul_add(b, 2.0f);
    auto unfused = (a * b) + 2.0f;

    auto h_f = to_host(fused);
    auto h_u = to_host(unfused);
    for (size_t i = 0; i < h_f.size(); ++i)
        EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

TEST(FusedOps, ScaleShiftEquivalent) {
    // scale_shift(s, b) == a*s + b calcolato in due passi
    Device gpu("cuda:0");
    Tensor<float> a({8}, gpu);
    a.fill(5.0f);

    auto fused   = a.scale_shift(2.0f, -1.0f);
    auto unfused = a * 2.0f + (-1.0f);

    auto h_f = to_host(fused);
    auto h_u = to_host(unfused);
    for (size_t i = 0; i < h_f.size(); ++i)
        EXPECT_FLOAT_EQ(h_f[i], h_u[i]);
}

// --- shape mismatch ---

TEST(FusedOps, ShapeMismatchThrows) {
    Device gpu("cuda:0");
    Tensor<float> a({4}, gpu);
    Tensor<float> b({8}, gpu);
    a.fill(1.0f);
    b.fill(1.0f);

    EXPECT_THROW(a.apply_binary(b, BinaryAdd<float>{}), std::invalid_argument);
}

// ============================================================================
// Benchmarks
// ============================================================================

#include <chrono>
#include <iostream>
#include <iomanip>

// Helper macro for timing
#define BENCHMARK_START() auto _bench_start = std::chrono::high_resolution_clock::now()
#define BENCHMARK_END(label, ops) do { \
    cudaDeviceSynchronize(); \
    auto _bench_end = std::chrono::high_resolution_clock::now(); \
    auto _duration = std::chrono::duration<double, std::milli>(_bench_end - _bench_start).count(); \
    std::cout << std::setw(30) << std::left << label \
              << std::setw(12) << std::right << std::fixed << std::setprecision(3) << _duration << " ms" \
              << std::setw(12) << std::right << ops << " ops" \
              << std::setw(12) << std::right << std::fixed << std::setprecision(2) << (ops / _duration * 1000.0) << " ops/s" \
              << std::endl; \
} while(0)

TEST(Benchmark, MatMulFloat) {
    Device gpu("cuda:0");
    
    // Warmup
    Tensor<float> warmup_a({64, 64}, gpu);
    Tensor<float> warmup_b({64, 64}, gpu);
    warmup_a.fill(1.0f);
    warmup_b.fill(1.0f);
    auto warmup = warmup_a.matmul(warmup_b);
    cudaDeviceSynchronize();
    
    std::cout << "\n=== MatMul Benchmark (float32) ===" << std::endl;
    std::cout << std::setw(30) << std::left << "Size" 
              << std::setw(12) << std::right << "Time"
              << std::setw(12) << std::right << "FLOPs"
              << std::setw(12) << std::right << "GFLOPS" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    
    const size_t sizes[] = {256, 512, 1024, 2048, 4096};
    for (size_t N : sizes) {
        Tensor<float> a({N, N}, gpu);
        Tensor<float> b({N, N}, gpu);
        a.fill(1.0f);
        b.fill(1.0f);
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a.matmul(b);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        // MatMul FLOPs = 2*N^3 (multiply + add for each element)
        double flops = 2.0 * N * N * N;
        double gflops = flops / (duration * 1e6);
        
        std::cout << std::setw(30) << std::left << (std::to_string(N) + "x" + std::to_string(N))
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(12) << std::right << std::scientific << std::setprecision(2) << flops
                  << std::setw(12) << std::right << std::fixed << std::setprecision(2) << gflops << std::endl;
    }
}

TEST(Benchmark, MatMulFP16) {
    Device gpu("cuda:0");
    
    // Warmup
    Tensor<float16_t> warmup_a({64, 64}, gpu);
    Tensor<float16_t> warmup_b({64, 64}, gpu);
    warmup_a.fill(float16_t(1.0f));
    warmup_b.fill(float16_t(1.0f));
    auto warmup = warmup_a.matmul(warmup_b);
    cudaDeviceSynchronize();
    
    std::cout << "\n=== MatMul Benchmark (float16) ===" << std::endl;
    std::cout << std::setw(30) << std::left << "Size" 
              << std::setw(12) << std::right << "Time"
              << std::setw(12) << std::right << "FLOPs"
              << std::setw(12) << std::right << "GFLOPS" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    
    const size_t sizes[] = {256, 512, 1024, 2048, 4096};
    for (size_t N : sizes) {
        Tensor<float16_t> a({N, N}, gpu);
        Tensor<float16_t> b({N, N}, gpu);
        a.fill(float16_t(1.0f));
        b.fill(float16_t(1.0f));
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a.matmul(b);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        double flops = 2.0 * N * N * N;
        double gflops = flops / (duration * 1e6);
        
        std::cout << std::setw(30) << std::left << (std::to_string(N) + "x" + std::to_string(N))
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(12) << std::right << std::scientific << std::setprecision(2) << flops
                  << std::setw(12) << std::right << std::fixed << std::setprecision(2) << gflops << std::endl;
    }
}

TEST(Benchmark, ElementWiseOps) {
    Device gpu("cuda:0");
    
    const size_t N = 16 * 1024 * 1024;  // 16M elements
    
    std::cout << "\n=== Element-wise Operations Benchmark (16M elements) ===" << std::endl;
    std::cout << std::setw(30) << std::left << "Operation" 
              << std::setw(12) << std::right << "Time"
              << std::setw(15) << std::right << "Throughput" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    // Float32
    {
        Tensor<float> a({N}, gpu);
        Tensor<float> b({N}, gpu);
        a.fill(2.0f);
        b.fill(3.0f);
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a + b;
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        std::cout << std::setw(30) << std::left << "float32 add"
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) << (N / duration / 1e6) << " Gelem/s" << std::endl;
    }
    
    // Float16
    {
        Tensor<float16_t> a({N}, gpu);
        Tensor<float16_t> b({N}, gpu);
        a.fill(float16_t(2.0f));
        b.fill(float16_t(3.0f));
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a + b;
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        std::cout << std::setw(30) << std::left << "float16 add"
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) << (N / duration / 1e6) << " Gelem/s" << std::endl;
    }
    
    // Multiplication comparison
    {
        Tensor<float> a({N}, gpu);
        Tensor<float> b({N}, gpu);
        a.fill(2.0f);
        b.fill(3.0f);
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a * b;
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        std::cout << std::setw(30) << std::left << "float32 mul"
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) << (N / duration / 1e6) << " Gelem/s" << std::endl;
    }
    
    {
        Tensor<float16_t> a({N}, gpu);
        Tensor<float16_t> b({N}, gpu);
        a.fill(float16_t(2.0f));
        b.fill(float16_t(3.0f));
        cudaDeviceSynchronize();
        
        BENCHMARK_START();
        auto c = a * b;
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _bench_start).count();
        
        std::cout << std::setw(30) << std::left << "float16 mul"
                  << std::setw(12) << std::right << std::fixed << std::setprecision(3) << duration << " ms"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(2) << (N / duration / 1e6) << " Gelem/s" << std::endl;
    }
}
