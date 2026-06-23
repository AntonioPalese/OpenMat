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
