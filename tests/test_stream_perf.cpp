#include "test_helpers.h"
#include "stream.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double elapsed_ms(Clock::time_point t0) {
    return Ms(Clock::now() - t0).count();
}

static void warmup(const Device& gpu) {
    Tensor<float> w({64, 64}, gpu);
    w.fill(1.0f);
    auto r = w + w;
    CUDA_CALL(cudaDeviceSynchronize());
}

static void print_header(const std::string& title) {
    std::cout << "\n+-- " << title << "\n"
              << "|   " << std::setw(32) << std::left << "variant"
              << std::setw(10) << std::right << "ms"
              << std::setw(10) << "speedup\n"
              << "+" << std::string(54, '-') << "\n";
}

static void print_row(const std::string& label, double ms, double ref_ms) {
    std::cout << "|   " << std::setw(32) << std::left << label
              << std::setw(9)  << std::right << std::fixed << std::setprecision(2) << ms << " ms"
              << std::setw(8)  << std::setprecision(2) << (ref_ms / ms) << "x\n";
}

// ── 1. Single op: sync vs default_stream vs explicit stream ──────────────────
//
// For a single op the three variants execute the same kernel. Any timing
// difference is host-side overhead only. The "speedup" column should be ~1x.

TEST(StreamPerf, SingleOpLatency) {
    Device gpu("cuda:0");
    warmup(gpu);

    constexpr int    ITERS = 200;
    constexpr size_t N     = 4 * 1024 * 1024;  // 16 MB

    Tensor<float> a({N}, gpu); a.fill(1.0f);
    Tensor<float> b({N}, gpu); b.fill(2.0f);
    CUDA_CALL(cudaDeviceSynchronize());

    // baseline: operator+ (internally uses default_stream + sync)
    auto t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto r = a + b;
        CUDA_CALL(cudaDeviceSynchronize());
    }
    double ms_sync = elapsed_ms(t0) / ITERS;

    // explicit default_stream()
    t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto r = a.add(b, om::Stream::default_stream());
        CUDA_CALL(cudaDeviceSynchronize());
    }
    double ms_default = elapsed_ms(t0) / ITERS;

    // explicit non-null stream, sync after each
    om::Stream s;
    t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        auto r = a.add(b, s);
        s.synchronize();
    }
    double ms_stream = elapsed_ms(t0) / ITERS;

    print_header("Single op (16M add, " + std::to_string(ITERS) + " iters, sync-after-each)");
    print_row("operator+ (sync)",        ms_sync,    ms_sync);
    print_row("add(default_stream())",   ms_default, ms_sync);
    print_row("add(Stream s) + sync",    ms_stream,  ms_sync);
    std::cout << "+" << std::string(54, '-') << "\n";

    // All three should be within 2x — they run the same kernel
    EXPECT_LT(ms_stream  / ms_sync, 2.0) << "stream variant unexpectedly slow";
    EXPECT_LT(ms_default / ms_sync, 2.0) << "default_stream variant unexpectedly slow";
}

// ── 2. Sequential chain: one sync at end vs sync after every op ──────────────
//
// Sync version:  launch -> sync -> launch -> sync -> ... (N host round-trips)
// Stream version: launch -> launch -> ... -> sync once  (1 host round-trip)
//
// The GPU does identical work; the gain is eliminating N-1 host/device stalls.

TEST(StreamPerf, SequentialChain) {
    Device gpu("cuda:0");
    warmup(gpu);

    constexpr int    DEPTH = 100;
    constexpr size_t N     = 2 * 1024 * 1024;  // 8 MB

    Tensor<float> a({N}, gpu); a.fill(1.0f);
    Tensor<float> b({N}, gpu); b.fill(1.0f);
    CUDA_CALL(cudaDeviceSynchronize());

    // sync after every op
    auto t0 = Clock::now();
    {
        Tensor<float> cur = a.add(b);
        CUDA_CALL(cudaDeviceSynchronize());
        for (int i = 1; i < DEPTH; ++i) {
            cur = cur.add(b);
            CUDA_CALL(cudaDeviceSynchronize());
        }
    }
    double ms_sync = elapsed_ms(t0);

    // single stream, single sync at end
    om::Stream s;
    t0 = Clock::now();
    {
        Tensor<float> cur = a.add(b, s);
        for (int i = 1; i < DEPTH; ++i)
            cur = cur.add(b, s);
        s.synchronize();
    }
    double ms_stream = elapsed_ms(t0);

    print_header("Sequential chain (" + std::to_string(DEPTH) + " adds, 8MB, single run)");
    print_row("sync after each op",    ms_sync,   ms_sync);
    print_row("stream + 1 sync",       ms_stream, ms_sync);
    std::cout << "+" << std::string(54, '-') << "\n";

    EXPECT_LT(ms_stream, ms_sync)
        << "stream chain should be faster than sync-after-each";
}

// ── 3. Parallel fan-out: K independent ops on K streams vs sequential ─────────
//
// Sequential:  op0 -> sync -> op1 -> sync -> ... (K serialised kernels)
// Parallel:    op0(s0) | op1(s1) | ... | sync-all (GPU runs them concurrently)
//
// The speedup depends on how many independent kernels the GPU can co-schedule.

TEST(StreamPerf, ParallelFanOut) {
    Device gpu("cuda:0");
    warmup(gpu);

    constexpr size_t N = 1 * 1024 * 1024;  // 4 MB per tensor

    print_header("Fan-out: K independent muls (4MB each)");

    for (int K : {2, 4, 8, 16}) {
        std::vector<Tensor<float>> inputs;
        inputs.reserve(K);
        for (int i = 0; i < K; ++i) {
            inputs.emplace_back(std::vector<size_t>{N}, gpu);
            inputs.back().fill(static_cast<float>(i + 1));
        }
        CUDA_CALL(cudaDeviceSynchronize());

        // sequential
        auto t0 = Clock::now();
        for (int i = 0; i < K; ++i) {
            auto r = inputs[i].mul(2.0f);
            CUDA_CALL(cudaDeviceSynchronize());
        }
        double ms_seq = elapsed_ms(t0);

        // K parallel streams
        std::vector<om::Stream> streams;
        streams.reserve(K);
        for (int i = 0; i < K; ++i) streams.emplace_back();

        t0 = Clock::now();
        std::vector<Tensor<float>> results;
        results.reserve(K);
        for (int i = 0; i < K; ++i)
            results.push_back(inputs[i].mul(2.0f, streams[i]));
        for (auto& st : streams) st.synchronize();
        double ms_par = elapsed_ms(t0);

        auto v = to_host(results[0]);
        EXPECT_FLOAT_EQ(v[0], 2.0f);

        std::string lseq = "K=" + std::to_string(K) + " sequential";
        std::string lpar = "K=" + std::to_string(K) + " parallel streams";
        print_row(lseq, ms_seq, ms_seq);
        print_row(lpar, ms_par, ms_seq);

        EXPECT_LE(ms_par, ms_seq * 1.2)
            << "parallel should not be slower than sequential for K=" << K;
    }
    std::cout << "+" << std::string(54, '-') << "\n";
}

// ── 4. Compute + transfer overlap ─────────────────────────────────────────────
//
// Overlaps H2D transfers on a copy stream with independent compute on a
// compute stream. On GPUs with a dedicated DMA engine both can run together.
//
// Serialised:  transfer -> sync -> compute -> sync  (transfer + compute time)
// Overlapped:  transfer || compute -> sync          (~max(transfer, compute))

TEST(StreamPerf, ComputeTransferOverlap) {
    Device gpu("cuda:0");
    warmup(gpu);

    constexpr int    ROUNDS = 20;
    constexpr size_t N      = 4 * 1024 * 1024;  // 16 MB

    std::vector<std::vector<float>> h_data(ROUNDS, std::vector<float>(N, 1.0f));
    Tensor<float> compute_src({N}, gpu); compute_src.fill(3.0f);
    CUDA_CALL(cudaDeviceSynchronize());

    // serialised
    auto t0 = Clock::now();
    for (int r = 0; r < ROUNDS; ++r) {
        auto uploaded = Tensor<float>::from_vector(h_data[r], {N}, gpu);
        CUDA_CALL(cudaDeviceSynchronize());
        auto computed = compute_src + uploaded;
        CUDA_CALL(cudaDeviceSynchronize());
    }
    double ms_serial = elapsed_ms(t0);

    // overlapped
    om::Stream stream_copy, stream_compute;
    t0 = Clock::now();
    for (int r = 0; r < ROUNDS; ++r) {
        auto uploaded = Tensor<float>::from_vector(h_data[r], {N}, gpu, stream_copy);
        auto computed = compute_src.mul(2.0f, stream_compute);
    }
    stream_copy.synchronize();
    stream_compute.synchronize();
    double ms_overlap = elapsed_ms(t0);

    print_header("Compute+transfer overlap (" + std::to_string(ROUNDS) + " rounds, 16MB)");
    print_row("serialised (H2D then compute)", ms_serial,  ms_serial);
    print_row("overlapped (2 streams)",        ms_overlap, ms_serial);
    std::cout << "+" << std::string(54, '-') << "\n";

    EXPECT_LE(ms_overlap, ms_serial * 1.1)
        << "overlapped should not be slower than serialised";
}

// ── 5. Stream reuse vs new stream per call ────────────────────────────────────
//
// cudaStreamCreate has driver overhead (~10-100 us). This test quantifies how
// much it costs to create a new stream per operation vs reusing one.

TEST(StreamPerf, StreamCreationOverhead) {
    Device gpu("cuda:0");
    warmup(gpu);

    constexpr int    ITERS = 1000;
    constexpr size_t N     = 64 * 1024;  // 256 KB

    Tensor<float> a({N}, gpu); a.fill(1.0f);
    CUDA_CALL(cudaDeviceSynchronize());

    // reuse one stream
    auto t0 = Clock::now();
    {
        om::Stream s;
        for (int i = 0; i < ITERS; ++i) {
            auto r = a.mul(2.0f, s);
            s.synchronize();
        }
    }
    double ms_reuse = elapsed_ms(t0) / ITERS;

    // create a new stream per iteration
    t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        om::Stream s;
        auto r = a.mul(2.0f, s);
        s.synchronize();
    }
    double ms_new = elapsed_ms(t0) / ITERS;

    print_header("Stream creation overhead (" + std::to_string(ITERS) + " iters, 256KB mul)");
    print_row("reuse stream",          ms_reuse, ms_reuse);
    print_row("new stream per call",   ms_new,   ms_reuse);
    std::cout << "+" << std::string(54, '-') << "\n";

    SUCCEED();
}
