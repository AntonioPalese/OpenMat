#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstddef>

#define CUDA_CHECK                                                          \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("[CUDA ERROR CHECK] at ") +                     \
                __FILE__ + ":" + std::to_string(__LINE__) + " → " +        \
                cudaGetErrorString(err));                                   \
        }                                                                   \
    } while (0)

#define CUDA_CALL(expr)                                                     \
    do {                                                                    \
        cudaError_t err = (expr);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("[CUDA ERROR CALL] '") + #expr + "' at " +     \
                __FILE__ + ":" + std::to_string(__LINE__) + " → " +        \
                cudaGetErrorString(err));                                   \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Debug launch checking
//
// cudaGetLastError() right after a launch only sees *synchronous* launch
// errors (bad grid/block configuration, bad arguments). An out-of-bounds
// access inside the kernel is asynchronous: it surfaces at the next
// synchronization point, which on a non-default stream can be arbitrarily far
// from the kernel that caused it — typically as an "illegal memory access"
// reported by an unrelated call.
//
// CUDA_CHECK_LAUNCH(kernel_name, stream) always performs the synchronous
// check, and in debug mode additionally forces a cudaStreamSynchronize() on
// the launch stream, so the asynchronous error is reported with the name of
// the kernel and the site that launched it.
//
// Enabling debug mode:
//   * runtime  — export OPENMAT_DEBUG_SYNC=1     (also accepts true/yes/on)
//   * build    — cmake -DOM_DEBUG_SYNC=ON        (on unless OPENMAT_DEBUG_SYNC=0)
//
// Building with -DOM_NO_DEBUG_SYNC=ON compiles the forced synchronization out
// entirely; only the synchronous check remains.
//
// The implementation deliberately lives in one translation unit inside the
// library (src/cuda_debug.cpp) rather than being inline here: both switches are
// preprocessor-conditional, and an inline definition would give a consumer that
// compiles with different settings a second, conflicting definition — an ODR
// violation the linker resolves by silently picking one. As a non-inline
// function the behaviour is fixed by how the library was built, and the extra
// call is nothing against the microseconds of a kernel launch.
//
// Forced synchronization serializes the very overlap streams exist for, so it
// is a diagnostic mode, never a default. Note that after an illegal access the
// CUDA context is unusable; the point is to learn *where* it happened.
// ─────────────────────────────────────────────────────────────────────────────

namespace om {
namespace detail {

    // True when forced post-launch synchronization is active. Resolved once,
    // on first use, from OPENMAT_DEBUG_SYNC and the build-time default.
    bool debug_sync_launches();

    // Called immediately after a launch. `stream` is the stream the kernel was
    // enqueued on — nullptr for the default stream. Throws std::runtime_error
    // naming the kernel and the call site.
    void check_launch(const char* kernel, const char* func,
                      const char* file, int line, cudaStream_t stream);

    // ─────────────────────────────────────────────────────────────────────
    // Grid extent limits
    //
    // gridDim.y and gridDim.z are capped at 65535 on every compute
    // capability; only gridDim.x goes up to 2^31-1. A launcher that maps a
    // tensor axis straight onto blockIdx.z therefore has a shape ceiling,
    // past which the launch fails *synchronously* with "invalid
    // configuration argument" — a message that says nothing about which
    // axis was too large, and which reads like a generic CUDA error.
    //
    // Launchers check the grid before launching and fall back to a flat 1-D
    // layout (the _nd kernels) when it does not fit.
    //
    // Takes size_t rather than a dim3 on purpose: dim3 members are unsigned
    // int, so building one first would truncate an extent above 2^32 and
    // could turn an over-large grid into a plausible small one.
    inline bool grid_fits(size_t gx, size_t gy, size_t gz)
    {
        constexpr size_t max_x  = 2147483647ull;   // 2^31 - 1
        constexpr size_t max_yz = 65535ull;
        return gx <= max_x && gy <= max_yz && gz <= max_yz;
    }

} // namespace detail
} // namespace om

#define CUDA_CHECK_LAUNCH(KERNEL_NAME, STREAM)                              \
    ::om::detail::check_launch((KERNEL_NAME), __PRETTY_FUNCTION__,          \
                               __FILE__, __LINE__, (STREAM))

inline bool is_device_pointer(const void* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

#if CUDART_VERSION >= 10000
    // Since CUDA 10, must check return status and memoryType
    if (err != cudaSuccess) return false;
    return attr.type == cudaMemoryTypeDevice;
#else
    // For CUDA < 10
    if (err != cudaSuccess) return false;
    return attr.memoryType == cudaMemoryTypeDevice;
#endif
}
