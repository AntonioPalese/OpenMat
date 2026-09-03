#include "test_helpers.h"
#include "host_pool.h"
#include "allocator.h"

#include <cstdint>
#include <vector>

using om::detail::HostPool;

namespace {
    bool is_aligned(const void* p, size_t a) {
        return (reinterpret_cast<uintptr_t>(p) % a) == 0;
    }
}

// The recycling tests are meaningless when the cache is off or too small to hold
// the block they use — OPENMAT_HOST_CACHE_BYTES=0 is a supported configuration
// (it restores the plain malloc/free behaviour), so skip rather than fail.
#define OM_REQUIRE_HOST_CACHE(bytes)                                          \
    do {                                                                      \
        if (HostPool::instance().capacity() < (bytes))                        \
            GTEST_SKIP() << "host block cache disabled or smaller than "      \
                         << (bytes) << " bytes";                              \
    } while (0)

// ── size classes ─────────────────────────────────────────────────────────────

TEST(HostPoolClasses, SmallRequestsShareTheMinimumBlock) {
    EXPECT_EQ(HostPool::size_class(1),   512u);
    EXPECT_EQ(HostPool::size_class(511), 512u);
    EXPECT_EQ(HostPool::size_class(512), 512u);
}

TEST(HostPoolClasses, EightClassesPerOctave) {
    // 2^20 .. 2^21 is stepped by 2^17.
    EXPECT_EQ(HostPool::size_class(1u << 20),           1u << 20);
    EXPECT_EQ(HostPool::size_class((1u << 20) + 1),     (1u << 20) + (1u << 17));
    EXPECT_EQ(HostPool::size_class((1u << 21) - 1),     1u << 21);
}

TEST(HostPoolClasses, OvershootStaysUnderOneEighth) {
    for (size_t n = 512; n < (1u << 24); n = n + n / 3 + 1) {
        const size_t cls = HostPool::size_class(n);
        EXPECT_GE(cls, n);
        EXPECT_LE(cls - n, n / 8) << "class " << cls << " for " << n;
    }
}

TEST(HostPoolClasses, HugeRequestThrowsInsteadOfWrappingAround) {
    EXPECT_THROW(HostPool::size_class(static_cast<size_t>(-1)), std::bad_alloc);
}

// ── recycling ────────────────────────────────────────────────────────────────

TEST(HostPoolRecycle, FreedBlockComesBack) {
    OM_REQUIRE_HOST_CACHE(1u << 20);
    om::CpuAllocator<float> alloc;
    const size_t n = 1u << 18;                  // 1 MB — past glibc's mmap threshold

    float* a = alloc.allocate(n);
    alloc.deallocate(a);
    float* b = alloc.allocate(n);

    EXPECT_EQ(a, b) << "the block should have been recycled, not re-mmap'd";
    alloc.deallocate(b);
}

TEST(HostPoolRecycle, RequestsInTheSameClassShareBlocks) {
    OM_REQUIRE_HOST_CACHE(1u << 20);
    om::CpuAllocator<char> alloc;

    char* a = alloc.allocate(1u << 20);
    alloc.deallocate(a);
    char* b = alloc.allocate((1u << 20) - 8);   // same size class
    EXPECT_EQ(a, b);
    alloc.deallocate(b);
}

TEST(HostPoolRecycle, CachedBytesTracksTheFreeList) {
    OM_REQUIRE_HOST_CACHE(1u << 18);
    HostPool& pool = HostPool::instance();
    pool.release_all();
    ASSERT_EQ(pool.cached_bytes(), 0u);

    om::CpuAllocator<float> alloc;
    const size_t n = 1u << 16;
    float* a = alloc.allocate(n);
    EXPECT_EQ(pool.cached_bytes(), 0u);         // live, not cached

    alloc.deallocate(a);
    EXPECT_EQ(pool.cached_bytes(), HostPool::size_class(n * sizeof(float)));

    pool.release_all();
    EXPECT_EQ(pool.cached_bytes(), 0u);
}

TEST(HostPoolRecycle, CacheStaysUnderItsCap) {
    HostPool& pool = HostPool::instance();
    pool.release_all();

    om::CpuAllocator<char> alloc;
    std::vector<char*> blocks;
    const size_t block = 8u << 20;
    const size_t count = pool.capacity() / block + 4;   // deliberately overshoot

    for (size_t i = 0; i < count; ++i)
        blocks.push_back(alloc.allocate(block));
    for (char* p : blocks)
        alloc.deallocate(p);

    EXPECT_LE(pool.cached_bytes(), pool.capacity());
    pool.release_all();
}

// ── allocator contract ───────────────────────────────────────────────────────

TEST(HostPoolContract, ZeroCountIsNullAndFreeingNullIsSafe) {
    om::CpuAllocator<float> alloc;
    EXPECT_EQ(alloc.allocate(0), nullptr);
    alloc.deallocate(nullptr);                  // must not crash
}

TEST(HostPoolContract, BlocksAreSixtyFourByteAligned) {
    om::CpuAllocator<float> alloc;
    for (size_t n : {1u, 100u, 4096u, 1u << 18}) {
        float* p = alloc.allocate(n);
        EXPECT_TRUE(is_aligned(p, 64)) << "n = " << n;
        alloc.deallocate(p);
    }
}

TEST(HostPoolContract, RecycledBlockIsFullyWritable) {
    om::CpuAllocator<float> alloc;
    const size_t n = 1u << 18;

    float* a = alloc.allocate(n);
    for (size_t i = 0; i < n; ++i) a[i] = static_cast<float>(i);
    alloc.deallocate(a);

    float* b = alloc.allocate(n);
    for (size_t i = 0; i < n; ++i) b[i] = -1.f;
    EXPECT_FLOAT_EQ(b[0],     -1.f);
    EXPECT_FLOAT_EQ(b[n - 1], -1.f);
    alloc.deallocate(b);
}

TEST(HostPoolContract, OverflowingCountThrows) {
    om::CpuAllocator<float> alloc;
    EXPECT_THROW(alloc.allocate(static_cast<size_t>(-1) / 2), std::bad_alloc);
}

// ── through a Tensor ─────────────────────────────────────────────────────────

TEST(HostPoolTensor, HostTensorsStillComputeCorrectly) {
    Device cpu("cpu:0");
    Tensor<float> a({64, 64}, cpu);
    Tensor<float> b({64, 64}, cpu);
    a.fill(2.f);
    b.fill(3.f);

    Tensor<float> c = a + b;
    EXPECT_FLOAT_EQ(c({0, 0}),   5.f);
    EXPECT_FLOAT_EQ(c({63, 63}), 5.f);
}

TEST(HostPoolTensor, RepeatedOpsDoNotGrowTheCacheWithoutBound) {
    HostPool& pool = HostPool::instance();
    pool.release_all();

    Device cpu("cpu:0");
    Tensor<float> a({256, 256}, cpu);
    Tensor<float> b({256, 256}, cpu);
    a.fill(1.f);
    b.fill(1.f);

    for (int i = 0; i < 100; ++i) {
        Tensor<float> c = a + b;
        EXPECT_FLOAT_EQ(c({0, 0}), 2.f);
    }

    // 100 iterations recycle the same block instead of accumulating 100 of them.
    EXPECT_LE(pool.cached_bytes(), 4u * HostPool::size_class(256 * 256 * sizeof(float)));
    pool.release_all();
}

// ── PinnedHostPool ────────────────────────────────────────────────────────────
// Page-locked counterpart of HostPool, backing PinnedCpuAllocator. Every test
// here touches the driver (cudaHostAlloc/cudaFreeHost), so — unlike the plain
// HostPool tests above — they all require a real device.

using om::detail::PinnedHostPool;

#define OM_REQUIRE_PINNED_CACHE(bytes)                                        \
    do {                                                                      \
        if (PinnedHostPool::instance().capacity() < (bytes))                  \
            GTEST_SKIP() << "pinned block cache disabled or smaller than "    \
                         << (bytes) << " bytes";                              \
    } while (0)

TEST(PinnedHostPoolRecycle, FreedBlockComesBack) {
    OM_REQUIRE_CUDA();
    OM_REQUIRE_PINNED_CACHE(1u << 20);
    om::PinnedCpuAllocator<float> alloc;
    const size_t n = 1u << 18;

    float* a = alloc.allocate(n);
    alloc.deallocate(a);
    float* b = alloc.allocate(n);

    EXPECT_EQ(a, b) << "the block should have been recycled, not re-pinned";
    alloc.deallocate(b);
}

TEST(PinnedHostPoolRecycle, CacheStaysUnderItsCap) {
    OM_REQUIRE_CUDA();
    PinnedHostPool& pool = PinnedHostPool::instance();
    pool.release_all();

    om::PinnedCpuAllocator<char> alloc;
    std::vector<char*> blocks;
    const size_t block = 4u << 20;
    const size_t count = pool.capacity() / block + 4;   // deliberately overshoot

    for (size_t i = 0; i < count; ++i)
        blocks.push_back(alloc.allocate(block));
    for (char* p : blocks)
        alloc.deallocate(p);

    EXPECT_LE(pool.cached_bytes(), pool.capacity());
    pool.release_all();
}

TEST(PinnedHostPoolContract, ZeroCountIsNullAndFreeingNullIsSafe) {
    OM_REQUIRE_CUDA();
    om::PinnedCpuAllocator<float> alloc;
    EXPECT_EQ(alloc.allocate(0), nullptr);
    alloc.deallocate(nullptr);                  // must not crash
}

TEST(PinnedHostPoolContract, BlockIsPageLockedAndWritable) {
    OM_REQUIRE_CUDA();
    om::PinnedCpuAllocator<float> alloc;
    const size_t n = 1024;

    float* p = alloc.allocate(n);
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);

    cudaPointerAttributes attrs{};
    CUDA_CALL(cudaPointerGetAttributes(&attrs, p));
    EXPECT_EQ(attrs.type, cudaMemoryTypeHost) << "allocate() must hand out page-locked memory";

    EXPECT_FLOAT_EQ(p[0], 0.f);
    EXPECT_FLOAT_EQ(p[n - 1], static_cast<float>(n - 1));
    alloc.deallocate(p);
}

// ── through a Tensor ─────────────────────────────────────────────────────────

TEST(PinnedHostPoolTensor, PinnedFactoryProducesAPinnedCpuTensor) {
    OM_REQUIRE_CUDA();
    auto t = Tensor<float>::pinned({64});
    EXPECT_EQ(t.device_type(), DEVICE_TYPE::CPU);
    EXPECT_TRUE(t.is_pinned());

    // Ordinary CPU allocation still isn't pinned.
    Tensor<float> plain({64}, Device("cpu:0"));
    EXPECT_FALSE(plain.is_pinned());
}

TEST(PinnedHostPoolTensor, PinnedTensorComputesCorrectly) {
    OM_REQUIRE_CUDA();
    auto a = Tensor<float>::pinned({64});
    Tensor<float> b({64}, Device("cpu:0"));
    a.fill(2.f);
    b.fill(3.f);

    Tensor<float> c = a + b;   // pinned memory is ordinary host-addressable memory
    EXPECT_FLOAT_EQ(c({0}), 5.f);
    EXPECT_FLOAT_EQ(c({63}), 5.f);
}

TEST(PinnedHostPoolTensor, D2HDestinationIsPinnedSyncAndAsync) {
    OM_REQUIRE_CUDA();
    Tensor<float> g({16}, Device("cuda:0"));
    g.fill(9.f);

    auto c1 = g.cpu();
    EXPECT_TRUE(c1.is_pinned());
    EXPECT_FLOAT_EQ(c1({0}), 9.f);

    om::Stream s;
    auto c2 = g.cpu(s);
    s.synchronize();
    EXPECT_TRUE(c2.is_pinned());
    EXPECT_FLOAT_EQ(c2({0}), 9.f);
}

TEST(PinnedHostPoolTensor, H2DSourceUnaffectedButCopyIsCorrect) {
    OM_REQUIRE_CUDA();
    // The H2D side has no automatic pinning (the source tensor already
    // exists), but Tensor::pinned() lets a caller opt in explicitly, and
    // either way the copy must still be correct.
    auto pinned_src = Tensor<float>::pinned({4});
    pinned_src({0}) = 1.f; pinned_src({1}) = 2.f; pinned_src({2}) = 3.f; pinned_src({3}) = 4.f;

    auto gpu_t = pinned_src.cuda();
    EXPECT_EQ(gpu_t.device_type(), DEVICE_TYPE::CUDA);
    auto back = to_host(gpu_t);
    EXPECT_FLOAT_EQ(back[0], 1.f);
    EXPECT_FLOAT_EQ(back[3], 4.f);
}
