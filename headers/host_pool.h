#pragma once

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace om {
namespace detail {

    // ─────────────────────────────────────────────────────────────────────────
    // Host block cache
    //
    // CpuAllocator used to hand every allocation straight to malloc/free.
    // Above glibc's MMAP_THRESHOLD (128 KB by default) that means a fresh mmap
    // per allocation and a munmap per free, so every out-of-place op faults in
    // its whole output buffer before it computes anything — 16384 page faults
    // for a 64 MB result. Measured, that is the entire gap against NumPy on
    // `add` at 16 M elements (19.4 ms → 4.5 ms, NumPy 6.1 ms) and the whole of
    // the D2H anomaly (18.2 ms → 1.14 ms, PyTorch 1.14 ms): cudaMemcpy into
    // never-touched pageable memory pays the same faults inside the copy.
    //
    // So blocks are recycled instead of being returned to the OS. Freeing
    // pushes the block onto a free list keyed by its size class; the next
    // allocation of that class gets it back with its pages still mapped. This
    // is what NumPy's block cache and PyTorch's CPU caching allocator do.
    //
    // Requests are rounded up to a size class — 8 classes per power of two, so
    // at most 12.5 % overshoot — to keep the number of lists bounded. Each
    // block carries a 64-byte header holding its class size, so deallocate()
    // knows which list a pointer belongs to without a side table, and the
    // pointer handed out is 64-byte aligned (malloc only guarantees 16).
    //
    // Process-wide, mutex-guarded, and capped: past the cap a freed block is
    // really freed. OPENMAT_HOST_CACHE_BYTES overrides the cap; 0 disables
    // recycling entirely and restores the old malloc/free behaviour.
    // ─────────────────────────────────────────────────────────────────────────
    class HostPool
    {
    public:
        static HostPool& instance()
        {
            // Deliberately never destroyed. A Tensor with static storage
            // duration — or anything else freeing host memory during static
            // destruction — would otherwise reach a pool that has already run
            // its destructor. The cached blocks are reclaimed by the OS at
            // exit like any other still-mapped page.
            static HostPool* pool = new HostPool();
            return *pool;
        }

        void* allocate(size_t bytes)
        {
            const size_t cls = size_class(bytes);

            if (m_Capacity != 0) {
                std::lock_guard<std::mutex> lock(m_Mutex);
                auto it = m_Free.find(cls);
                if (it != m_Free.end() && !it->second.empty()) {
                    void* base = it->second.back();
                    it->second.pop_back();
                    m_Cached -= cls;
                    return user_ptr(base);
                }
            }

            void* base = std::aligned_alloc(kAlign, kAlign + cls);
            if (!base) {
                // Out of memory while the cache is holding blocks of the wrong
                // size classes: give them back and try once more.
                release_all();
                base = std::aligned_alloc(kAlign, kAlign + cls);
                if (!base)
                    throw std::bad_alloc();
            }
            *static_cast<size_t*>(base) = cls;
            return user_ptr(base);
        }

        void deallocate(void* ptr)
        {
            if (!ptr)
                return;

            void* base = base_ptr(ptr);
            const size_t cls = *static_cast<size_t*>(base);

            if (m_Capacity != 0) {
                std::lock_guard<std::mutex> lock(m_Mutex);
                if (m_Cached + cls <= m_Capacity) {
                    m_Free[cls].push_back(base);
                    m_Cached += cls;
                    return;
                }
            }
            std::free(base);
        }

        // Returns every cached block to the OS. Live allocations are untouched.
        void release_all()
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            for (auto& entry : m_Free) {
                for (void* base : entry.second)
                    std::free(base);
                entry.second.clear();
            }
            m_Cached = 0;
        }

        size_t cached_bytes()
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            return m_Cached;
        }

        size_t capacity() const { return m_Capacity; }

        // Bytes actually reserved for a request of `bytes` (header excluded).
        static size_t size_class(size_t bytes)
        {
            if (bytes > (static_cast<size_t>(-1) >> 1))
                throw std::bad_alloc();          // would overflow the round-up
            if (bytes <= kMinBlock)
                return kMinBlock;

            // 8 classes per octave: round up to a multiple of 2^(p-3), which
            // is never smaller than kAlign because p ≥ 9 here.
            const int    p    = floor_log2(bytes);
            const size_t step = static_cast<size_t>(1) << (p - kSubBits);
            return (bytes + step - 1) & ~(step - 1);
        }

    private:
        static constexpr size_t kAlign       = 64;             // header size too
        static constexpr size_t kMinBlock    = 512;
        static constexpr int    kSubBits     = 3;              // 2^3 classes per octave
        static constexpr size_t kDefaultCap  = 256ull << 20;   // 256 MB

        HostPool() : m_Capacity(env_capacity()) {}

        static size_t env_capacity()
        {
            const char* env = std::getenv("OPENMAT_HOST_CACHE_BYTES");
            if (!env || !*env)
                return kDefaultCap;
            char* end = nullptr;
            const unsigned long long v = std::strtoull(env, &end, 10);
            if (end == env)
                return kDefaultCap;
            return static_cast<size_t>(v);
        }

        static int floor_log2(size_t v)
        {
#if defined(__GNUC__) || defined(__clang__)
            return 63 - __builtin_clzll(static_cast<unsigned long long>(v));
#else
            int p = 0;
            while (v >>= 1) ++p;
            return p;
#endif
        }

        static void* user_ptr(void* base) { return static_cast<char*>(base) + kAlign; }
        static void* base_ptr(void* ptr)  { return static_cast<char*>(ptr) - kAlign; }

        std::mutex                                     m_Mutex;
        std::unordered_map<size_t, std::vector<void*>> m_Free;
        size_t                                         m_Cached   = 0;
        const size_t                                   m_Capacity;
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Pinned (page-locked) host block cache
    //
    // HostPool fixed every allocator-bound number except one: H2D/D2H transfer
    // into a pageable buffer still makes cudaMemcpyAsync synchronous, because
    // the driver has to stage the copy through its own internal pinned bounce
    // buffer rather than DMA'ing straight to/from the buffer HostPool handed
    // out. cudaHostAlloc/cudaFreeHost page-lock memory so that staging step is
    // skipped — see benchmark_report.md §3 and stream_perf_report.md.
    //
    // Page-locking is a page-table operation, not a malloc — a bare
    // cudaHostAlloc runs one to two orders of magnitude slower than the
    // equivalent pageable allocation. So recycling matters even more here than
    // in HostPool: without it, a repeated transfer of the same shape would pay
    // that cost on every call. Same size-class scheme as HostPool (reused via
    // HostPool::size_class, so the two can never drift apart); a smaller
    // default cap, since pinned pages are a scarcer resource the OS cannot
    // reclaim under memory pressure the way it reclaims ordinary pages.
    //
    // Deliberately not the allocator behind ordinary CPU tensors: nothing here
    // decides on its own which host tensors will cross the bus. This pool only
    // ever backs a PinnedCpuAllocator (headers/allocator.h), and a Tensor gets
    // one of those only when a caller explicitly asks (Tensor::pinned()) or
    // when Tensor::to() allocates the destination of a device-to-host copy,
    // where the answer is not a guess — that exact buffer is a transfer
    // destination right now. Both call sites already require a working CUDA
    // driver, so unlike HostPool this pool is never touched by the CPU-only
    // build/test job — it still has to compile there (against the stub
    // libcuda.so the cpu CI job links against), but nothing exercises it.
    //
    // One more difference from HostPool: its cached blocks are plain malloc,
    // invisible to compute-sanitizer's CUDA allocation tracking, so a
    // never-destroyed singleton is a non-issue for --leak-check. cudaHostAlloc
    // blocks are tracked, so the same design here would flag every cached (not
    // live) block as leaked at process exit — which is what the CI gpu job's
    // `--error-exitcode 1` treats as a hard failure. The constructor registers
    // an atexit hook that empties the free list on the way out. This does not
    // reintroduce the destruction-order hazard the comment above is guarding
    // against: it only calls release_all(), never deletes `pool`, so the
    // singleton itself is exactly as immortal as before — a deallocate() that
    // lands after this hook runs (from a Tensor with static storage duration,
    // say) still finds a live pool and simply repopulates its free list.
    // ─────────────────────────────────────────────────────────────────────────
    class PinnedHostPool
    {
    public:
        static PinnedHostPool& instance()
        {
            // Deliberately never destroyed, for the same reason as HostPool::
            // a Tensor with static storage duration must not outlive the pool
            // it frees pinned blocks into.
            static PinnedHostPool* pool = new PinnedHostPool();
            return *pool;
        }

        void* allocate(size_t bytes)
        {
            const size_t cls = HostPool::size_class(bytes);

            if (m_Capacity != 0) {
                std::lock_guard<std::mutex> lock(m_Mutex);
                auto it = m_Free.find(cls);
                if (it != m_Free.end() && !it->second.empty()) {
                    void* base = it->second.back();
                    it->second.pop_back();
                    m_Cached -= cls;
                    return user_ptr(base);
                }
            }

            void* base = alloc_block(cls);
            if (!base) {
                // Out of pinned memory while the cache holds blocks of the
                // wrong size classes: give them back and try once more.
                release_all();
                base = alloc_block(cls);
                if (!base)
                    throw std::bad_alloc();
            }
            *static_cast<size_t*>(base) = cls;
            return user_ptr(base);
        }

        void deallocate(void* ptr)
        {
            if (!ptr)
                return;

            void* base = base_ptr(ptr);
            const size_t cls = *static_cast<size_t*>(base);

            if (m_Capacity != 0) {
                std::lock_guard<std::mutex> lock(m_Mutex);
                if (m_Cached + cls <= m_Capacity) {
                    m_Free[cls].push_back(base);
                    m_Cached += cls;
                    return;
                }
            }
            cudaFreeHost(base);
        }

        // Returns every cached block to the driver. Live allocations untouched.
        void release_all()
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            for (auto& entry : m_Free) {
                for (void* base : entry.second)
                    cudaFreeHost(base);
                entry.second.clear();
            }
            m_Cached = 0;
        }

        size_t cached_bytes()
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            return m_Cached;
        }

        size_t capacity() const { return m_Capacity; }

    private:
        static constexpr size_t kAlign      = 64;             // header size too
        static constexpr size_t kDefaultCap = 64ull << 20;     // 64 MB

        PinnedHostPool() : m_Capacity(env_capacity())
        {
            std::atexit(&PinnedHostPool::release_cached_at_exit);
        }

        static void release_cached_at_exit()
        {
            instance().release_all();
        }

        static void* alloc_block(size_t cls)
        {
            void* base = nullptr;
            // Errors here are reported to the caller (allocate() retries once,
            // then throws) rather than via CUDA_CALL: a missing driver — the
            // cpu CI job — must never reach this function in the first place
            // (see the class comment), so throwing std::runtime_error instead
            // of pulling in cuda_defines.cuh's CUDA_CALL keeps this header
            // free of a dependency the HostPool half of the file doesn't need.
            cudaError_t err = cudaHostAlloc(&base, kAlign + cls, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                (void)cudaGetLastError();   // clear the sticky error, mirrors CUDA_CALL sites
                return nullptr;
            }
            return base;
        }

        static size_t env_capacity()
        {
            const char* env = std::getenv("OPENMAT_PINNED_CACHE_BYTES");
            if (!env || !*env)
                return kDefaultCap;
            char* end = nullptr;
            const unsigned long long v = std::strtoull(env, &end, 10);
            if (end == env)
                return kDefaultCap;
            return static_cast<size_t>(v);
        }

        static void* user_ptr(void* base) { return static_cast<char*>(base) + kAlign; }
        static void* base_ptr(void* ptr)  { return static_cast<char*>(ptr) - kAlign; }

        std::mutex                                     m_Mutex;
        std::unordered_map<size_t, std::vector<void*>> m_Free;
        size_t                                         m_Cached   = 0;
        const size_t                                   m_Capacity;
    };

} // namespace detail
} // namespace om
