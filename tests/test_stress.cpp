#include "test_helpers.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>

#define BENCH_START() auto _t0 = std::chrono::high_resolution_clock::now()
#define BENCH_MS() (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _t0).count())

// ── helpers ───────────────────────────────────────────────────────────────────

static void gpu_sync() { CUDA_CALL(cudaDeviceSynchronize()); }

// ── 1. Allocator pressure: many alloc/free cycles ────────────────────────────
// Allocates N_ITERS tensors one-by-one (each immediately freed at end of scope)
// and N_BATCH at once to stress the cudaMallocAsync pool.

TEST(Stress, AllocatorSequential) {
    Device gpu("cuda:0");
    constexpr int N_ITERS = 500;
    constexpr size_t ELEMS = 1024 * 1024;  // 4 MB each

    BENCH_START();
    for (int i = 0; i < N_ITERS; ++i) {
        Tensor<float> t({ELEMS}, gpu);
        t.fill(1.0f);
        gpu_sync();
    }
    double ms = BENCH_MS();
    std::cout << "\n[AllocatorSequential] " << N_ITERS << " x 4MB alloc+fill+free: "
              << std::fixed << std::setprecision(1) << ms << " ms  ("
              << std::setprecision(2) << (ms / N_ITERS) << " ms/iter)\n";
    SUCCEED();
}

TEST(Stress, AllocatorBatch) {
    Device gpu("cuda:0");
    constexpr int N_BATCH = 64;
    constexpr size_t ELEMS = 512 * 1024;  // 2 MB each

    BENCH_START();
    {
        std::vector<Tensor<float>> tensors;
        tensors.reserve(N_BATCH);
        for (int i = 0; i < N_BATCH; ++i) {
            tensors.emplace_back(std::vector<size_t>{ELEMS}, gpu);
            tensors.back().fill(static_cast<float>(i));
        }
        gpu_sync();
    }  // all freed here
    double ms = BENCH_MS();
    std::cout << "[AllocatorBatch]      " << N_BATCH << " tensors live simultaneously (2MB each): "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
    SUCCEED();
}

// ── 2. Sustained element-wise throughput ─────────────────────────────────────
// Runs 1000 iterations of add+mul to ensure no memory leak or error accumulation.

TEST(Stress, SustainedElementWise) {
    Device gpu("cuda:0");
    constexpr int N_ITERS = 1000;
    constexpr size_t ELEMS = 4 * 1024 * 1024;  // 16 MB

    Tensor<float> a({ELEMS}, gpu); a.fill(1.0f);
    Tensor<float> b({ELEMS}, gpu); b.fill(2.0f);
    gpu_sync();

    BENCH_START();
    for (int i = 0; i < N_ITERS; ++i) {
        auto c = a + b;
        auto d = c * b;
        (void)d;
    }
    gpu_sync();
    double ms = BENCH_MS();
    double gb_s = (N_ITERS * 3.0 * ELEMS * sizeof(float)) / (ms * 1e6);
    std::cout << "[SustainedElemWise]   " << N_ITERS << " x (add+mul) on 16M floats: "
              << std::fixed << std::setprecision(1) << ms << " ms  ("
              << std::setprecision(1) << gb_s << " GB/s effective)\n";
    SUCCEED();
}

// ── 3. Stream concurrency ─────────────────────────────────────────────────────
// Fires independent work on N_STREAMS streams simultaneously, then syncs all.

TEST(Stress, MultiStreamConcurrency) {
    Device gpu("cuda:0");
    constexpr int N_STREAMS = 8;
    constexpr size_t ELEMS = 2 * 1024 * 1024;

    std::vector<om::Stream> streams;
    streams.reserve(N_STREAMS);
    for (int i = 0; i < N_STREAMS; ++i)
        streams.emplace_back();

    std::vector<Tensor<float>> inputs;
    inputs.reserve(N_STREAMS);
    for (int i = 0; i < N_STREAMS; ++i) {
        inputs.emplace_back(std::vector<size_t>{ELEMS}, gpu);
        inputs.back().fill(static_cast<float>(i + 1));
    }
    gpu_sync();

    BENCH_START();
    std::vector<Tensor<float>> outputs;
    outputs.reserve(N_STREAMS);
    for (int i = 0; i < N_STREAMS; ++i)
        outputs.push_back(inputs[i].mul(2.0f, streams[i]));

    for (auto& s : streams) s.synchronize();
    double ms = BENCH_MS();

    // Verify correctness for stream 0: all elements should be 2.0
    auto v = to_host(outputs[0]);
    ASSERT_EQ(v.size(), ELEMS);
    EXPECT_FLOAT_EQ(v[0], 2.0f);
    EXPECT_FLOAT_EQ(v[ELEMS - 1], 2.0f);

    std::cout << "[MultiStreamConc]     " << N_STREAMS << " streams x 8MB mul in parallel: "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
}

// ── 4. Large single tensor ────────────────────────────────────────────────────
// Stress-tests single very large allocations and ops.

TEST(Stress, LargeTensor) {
    Device gpu("cuda:0");
    constexpr size_t ELEMS = 128 * 1024 * 1024;  // 512 MB

    Tensor<float> a({ELEMS}, gpu); a.fill(1.0f); gpu_sync();
    Tensor<float> b({ELEMS}, gpu); b.fill(2.0f); gpu_sync();

    BENCH_START();
    auto c = a + b; gpu_sync();
    double ms = BENCH_MS();
    double gb_s = (3.0 * ELEMS * sizeof(float)) / (ms * 1e6);

    auto v = to_host(c);
    EXPECT_FLOAT_EQ(v[0],        3.0f);
    EXPECT_FLOAT_EQ(v[ELEMS - 1], 3.0f);

    std::cout << "[LargeTensor]         512MB + 512MB add: "
              << std::fixed << std::setprecision(1) << ms << " ms  ("
              << std::setprecision(1) << gb_s << " GB/s)\n";
}

// ── 5. Op chain depth ─────────────────────────────────────────────────────────
// Long dependency chain: each result feeds the next op.
// Tests that the runtime handles many sequential kernel launches without
// resource exhaustion.

TEST(Stress, OpChainDepth) {
    Device gpu("cuda:0");
    constexpr int DEPTH = 200;
    constexpr size_t ELEMS = 1024 * 1024;

    Tensor<float> t({ELEMS}, gpu); t.fill(1.0f); gpu_sync();

    BENCH_START();
    Tensor<float> cur = std::move(t);
    for (int i = 0; i < DEPTH; ++i)
        cur = cur.add(1.0f);
    gpu_sync();
    double ms = BENCH_MS();

    auto v = to_host(cur);
    EXPECT_FLOAT_EQ(v[0], static_cast<float>(1 + DEPTH));

    std::cout << "[OpChainDepth]        " << DEPTH << "-deep add chain (4MB): "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
}

// ── 6. High-rank permute (hits _nd fallback) ──────────────────────────────────
// Rank 6 and rank 8 permutations exercise the flat-index fallback kernel path.

TEST(Stress, HighRankPermute) {
    Device gpu("cuda:0");

    // rank-6 tensor: [4,4,4,4,4,4] = 4096 elements
    {
        Tensor<float> t({4,4,4,4,4,4}, gpu); t.fill(1.0f); gpu_sync();
        BENCH_START();
        constexpr int ITERS = 1000;
        for (int i = 0; i < ITERS; ++i) {
            auto p = t.permute({5,4,3,2,1,0});
            (void)p;
        }
        gpu_sync();
        double ms = BENCH_MS();
        std::cout << "[HighRankPermute]     rank-6 [4^6] x" << ITERS << ": "
                  << std::fixed << std::setprecision(1) << ms << " ms\n";
    }

    // rank-8 tensor: [4,4,4,4,4,4,4,4] = 65536 elements
    {
        Tensor<float> t({4,4,4,4,4,4,4,4}, gpu); t.fill(2.0f); gpu_sync();
        auto p = t.permute({7,6,5,4,3,2,1,0});
        gpu_sync();
        auto v = to_host(p);
        EXPECT_FLOAT_EQ(v[0], 2.0f);
        std::cout << "[HighRankPermute]     rank-8 [4^8] reverse perm: OK\n";
    }
}

// ── 7. CPU stress: large sequential ops ───────────────────────────────────────
// Verifies CPU path doesn't degrade under load.

TEST(Stress, CPULarge) {
    Device cpu("cpu:0");
    constexpr size_t ELEMS = 8 * 1024 * 1024;  // 32 MB

    Tensor<float> a({ELEMS}, cpu); a.fill(3.0f);
    Tensor<float> b({ELEMS}, cpu); b.fill(2.0f);

    BENCH_START();
    auto c = a + b;
    auto d = c * b;
    double ms = BENCH_MS();

    EXPECT_FLOAT_EQ(c.view()[0], 5.0f);
    EXPECT_FLOAT_EQ(d.view()[0], 10.0f);

    std::cout << "[CPULarge]            32MB add+mul (CPU): "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
}

// ── 8. Repeated H2D / D2H transfers ───────────────────────────────────────────
// Measures async transfer bandwidth under repeated round-trips.

TEST(Stress, AsyncTransferBandwidth) {
    constexpr int N_ITERS = 100;
    constexpr size_t ELEMS = 16 * 1024 * 1024;  // 64 MB

    std::vector<float> host_data(ELEMS, 1.0f);
    Device gpu("cuda:0");
    om::Stream s;

    BENCH_START();
    for (int i = 0; i < N_ITERS; ++i) {
        auto t = Tensor<float>::from_vector(host_data, {ELEMS}, gpu, s);
        auto back = t.cpu(s);
        s.synchronize();
    }
    double ms = BENCH_MS();
    double gb_s = (N_ITERS * 2.0 * ELEMS * sizeof(float)) / (ms * 1e6);

    std::cout << "[AsyncTransfer]       " << N_ITERS << " x 64MB H2D+D2H round-trips: "
              << std::fixed << std::setprecision(1) << ms << " ms  ("
              << std::setprecision(1) << gb_s << " GB/s)\n";
    SUCCEED();
}
