#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <cuda_runtime.h>

// Keeps cudaMallocAsync's per-device pool from handing memory back to the OS.
//
// The pool's release threshold defaults to 0: at every synchronization
// (cudaDeviceSynchronize, cudaStreamSynchronize, ...) it trims all unused
// memory. The synchronous API synchronizes after every op, and a result
// dropped right away is freed back into the pool, which then gets trimmed. So
// the next allocation of the same size had to map fresh physical pages. On
// GB10 this cost ~150 µs per 64 MB op in a plain loop, and 3.2 ms once the
// caller also synchronized — more than the kernel itself. PyTorch never pays
// it, because its caching allocator never returns memory. Setting the
// threshold to "unlimited" gives the pool the same policy; the memory stays
// reusable by any later cudaMallocAsync on the device.
//
// OPENMAT_CUDA_POOL_RELEASE_BYTES overrides the threshold (bytes the pool may
// keep across a synchronize); 0 restores the CUDA default, for A/B.
//
// Out of line (not inline in allocator.inl) for the same reason as
// cuda_debug.cpp: one definition, one set of once-per-device state.
namespace om::detail
{
    namespace
    {
        uint64_t release_threshold()
        {
            const char* env = std::getenv("OPENMAT_CUDA_POOL_RELEASE_BYTES");
            if (!env || !*env)
                return UINT64_MAX;
            char* end = nullptr;
            const unsigned long long v = std::strtoull(env, &end, 10);
            return end == env ? UINT64_MAX : static_cast<uint64_t>(v);
        }

        void configure(int device)
        {
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) {
                cudaGetLastError();   // not fatal: the pool keeps CUDA's default policy
                return;
            }
            uint64_t threshold = release_threshold();
            if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess)
                cudaGetLastError();
        }
    }

    void ensure_device_pool_configured()
    {
#if CUDART_VERSION >= 11020
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cudaGetLastError();   // the allocation that follows reports the real error
            return;
        }

        // Lock-free after the first call on a device: one atomic load per
        // allocation. Devices past 63 take the mutex every time.
        static std::atomic<uint64_t> done_mask{0};
        if (device < 64 && ((done_mask.load(std::memory_order_acquire) >> device) & 1u))
            return;

        static std::mutex m;
        static std::set<int> done_high;
        std::lock_guard<std::mutex> lock(m);
        if (device < 64) {
            if ((done_mask.load(std::memory_order_relaxed) >> device) & 1u)
                return;
            configure(device);
            done_mask.fetch_or(uint64_t{1} << device, std::memory_order_release);
        } else if (done_high.insert(device).second) {
            configure(device);
        }
#endif
    }
}
