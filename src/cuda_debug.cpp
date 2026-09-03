// Implementation of the debug launch checking declared in cuda_defines.cuh.
//
// Kept in a single translation unit compiled into the library: OM_DEBUG_SYNC
// and OM_NO_DEBUG_SYNC are preprocessor switches, so an inline definition in
// the header would let a consumer built with different settings introduce a
// second, conflicting definition. See the comment in cuda_defines.cuh.

#include "cuda_defines.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Build-time default for the forced synchronization; the OPENMAT_DEBUG_SYNC
// environment variable overrides it in either direction.
#ifndef OM_DEBUG_SYNC
#define OM_DEBUG_SYNC 0
#endif

namespace om {
namespace detail {

namespace {

    // Parses a boolean environment variable. Absent or empty → default_value.
    bool env_flag(const char* name, bool default_value)
    {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return default_value;
        return !(std::strcmp(v, "0")     == 0 ||
                 std::strcmp(v, "false") == 0 ||
                 std::strcmp(v, "FALSE") == 0 ||
                 std::strcmp(v, "no")    == 0 ||
                 std::strcmp(v, "off")   == 0);
    }

    std::string launch_site(const char* kernel, const char* func,
                            const char* file, int line, cudaStream_t stream)
    {
        char stream_buf[32];
        std::snprintf(stream_buf, sizeof(stream_buf), "%p", static_cast<void*>(stream));

        return std::string("kernel '") + kernel + "' at " + file + ":" +
               std::to_string(line) + "\n  in " + func + "\n  on stream " +
               (stream == nullptr ? std::string("(default)")
                                  : std::string(stream_buf));
    }

} // namespace

    bool debug_sync_launches()
    {
        // Read once, on first use — a launch costs microseconds, this costs a
        // predicted branch on an already-initialized static.
        static const bool enabled = env_flag("OPENMAT_DEBUG_SYNC", OM_DEBUG_SYNC != 0);
        return enabled;
    }

    void check_launch(const char* kernel, const char* func,
                      const char* file, int line, cudaStream_t stream)
    {
        // Synchronous launch errors: bad grid/block configuration, bad args.
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA LAUNCH ERROR] ") +
                launch_site(kernel, func, file, line, stream) + "\n  → " +
                cudaGetErrorString(err));
        }

#ifndef OM_NO_DEBUG_SYNC
        if (!debug_sync_launches()) return;

        // Asynchronous errors: everything that happens inside the kernel.
        err = cudaStreamSynchronize(stream);
        if (err == cudaSuccess) err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA ASYNC ERROR] ") +
                launch_site(kernel, func, file, line, stream) + "\n  → " +
                cudaGetErrorString(err) +
                "\n  (caught by OPENMAT_DEBUG_SYNC forced synchronization; "
                "unset it to restore asynchronous execution)");
        }
#endif
    }

} // namespace detail
} // namespace om
