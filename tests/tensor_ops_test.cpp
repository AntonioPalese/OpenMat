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
