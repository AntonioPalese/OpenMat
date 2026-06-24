#include "test_helpers.h"
#include <chrono>
#include <iostream>
#include <iomanip>

#define BENCH_START() auto _t0 = std::chrono::high_resolution_clock::now()
#define BENCH_MS() (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _t0).count())

TEST(Benchmark, MatMulFloat) {
    Device gpu("cuda:0");
    { Tensor<float> w({64,64},gpu); w.fill(1.0f); w.matmul(w); cudaDeviceSynchronize(); }

    std::cout << "\n=== MatMul float32 ===\n"
              << std::setw(10) << "Size"
              << std::setw(12) << "ms"
              << std::setw(12) << "GFLOPS\n"
              << std::string(34, '-') << "\n";

    for (size_t N : {256u, 512u, 1024u, 2048u, 4096u}) {
        Tensor<float> a({N,N},gpu); a.fill(1.0f);
        Tensor<float> b({N,N},gpu); b.fill(1.0f);
        cudaDeviceSynchronize();
        BENCH_START();
        auto c = a.matmul(b); cudaDeviceSynchronize();
        double ms = BENCH_MS();
        double gflops = 2.0*N*N*N / (ms * 1e6);
        std::cout << std::setw(10) << (std::to_string(N)+"x"+std::to_string(N))
                  << std::setw(11) << std::fixed << std::setprecision(2) << ms << " ms"
                  << std::setw(11) << std::setprecision(2) << gflops << "\n";
    }
}

TEST(Benchmark, MatMulFP16) {
    Device gpu("cuda:0");
    { Tensor<float16_t> w({64,64},gpu); w.fill(float16_t(1.0f)); w.matmul(w); cudaDeviceSynchronize(); }

    std::cout << "\n=== MatMul float16 ===\n"
              << std::setw(10) << "Size"
              << std::setw(12) << "ms"
              << std::setw(12) << "GFLOPS\n"
              << std::string(34, '-') << "\n";

    for (size_t N : {256u, 512u, 1024u, 2048u, 4096u}) {
        Tensor<float16_t> a({N,N},gpu); a.fill(float16_t(1.0f));
        Tensor<float16_t> b({N,N},gpu); b.fill(float16_t(1.0f));
        cudaDeviceSynchronize();
        BENCH_START();
        auto c = a.matmul(b); cudaDeviceSynchronize();
        double ms = BENCH_MS();
        double gflops = 2.0*N*N*N / (ms * 1e6);
        std::cout << std::setw(10) << (std::to_string(N)+"x"+std::to_string(N))
                  << std::setw(11) << std::fixed << std::setprecision(2) << ms << " ms"
                  << std::setw(11) << std::setprecision(2) << gflops << "\n";
    }
}

TEST(Benchmark, ElementWiseOps) {
    Device gpu("cuda:0");
    const size_t N = 16 * 1024 * 1024;

    std::cout << "\n=== Element-wise ops (16M elements) ===\n"
              << std::setw(20) << "Op"
              << std::setw(12) << "ms"
              << std::setw(14) << "Gelem/s\n"
              << std::string(46, '-') << "\n";

    auto bench = [&](const std::string& label, auto fn) {
        cudaDeviceSynchronize();
        BENCH_START();
        auto r = fn(); cudaDeviceSynchronize();
        double ms = BENCH_MS();
        std::cout << std::setw(20) << label
                  << std::setw(11) << std::fixed << std::setprecision(2) << ms << " ms"
                  << std::setw(13) << std::setprecision(2) << (N / ms / 1e6) << "\n";
    };

    Tensor<float> af({N},gpu); af.fill(2.0f);
    Tensor<float> bf({N},gpu); bf.fill(3.0f);
    bench("float32 add",  [&]{ return af + bf; });
    bench("float32 mul",  [&]{ return af * bf; });
    bench("scale_shift",  [&]{ return af.scale_shift(2.0f, 1.0f); });
    bench("fused_add_mul",[&]{ return af.fused_add_mul(bf, 0.5f); });

    Tensor<float16_t> ah({N},gpu); ah.fill(float16_t(2.0f));
    Tensor<float16_t> bh({N},gpu); bh.fill(float16_t(3.0f));
    bench("float16 add",  [&]{ return ah + bh; });
    bench("float16 mul",  [&]{ return ah * bh; });
}
