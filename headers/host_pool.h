#pragma once

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

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

} // namespace detail
} // namespace om
