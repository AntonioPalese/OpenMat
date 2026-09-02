/**
 * C-ABI FFI layer for OpenMat.
 *
 * All functions use plain C types and extern "C" linkage so they are callable
 * from Python ctypes (or any other FFI) without name-mangling issues.
 *
 * Tensor handles are opaque void* that point to heap-allocated Tensor<T>.
 * The caller owns the lifetime: every om_*_create / om_*_copy must be matched
 * by exactly one om_*_destroy.
 *
 * Error handling: functions that can fail return an int (0 = success, non-zero
 * = error) and write a human-readable message to the provided errbuf.  Pointer-
 * returning functions return nullptr on failure.  No exception ever crosses the
 * ABI boundary — every entry point is wrapped in one of the OM_GUARD_* macros.
 *
 * The per-dtype surface lives in openmat_capi_impl.inc, which is included once
 * per supported element type; see the OM_T / OM_SFX pairs at the bottom.
 */

#include "tensor.cuh"
#include "mat_utils.h"
#include "stream.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Fills errbuf (if non-null, capacity errbuf_len) with a message.
void set_err(char* errbuf, int errbuf_len, const char* msg) {
    if (errbuf && errbuf_len > 0)
        std::snprintf(errbuf, errbuf_len, "%s", msg);
}

om::Device om_make_device(int on_cuda) {
    return om::Device(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
}

// Reference-counted stream.
//
// Memory a tensor got from a stream's pool must be freed on that same stream,
// so the stream has to outlive every tensor allocated on it.  Leaving that
// ordering to the binding language does not work: Python's cyclic collector
// finalizes a cycle's members (and everything reachable only from them) in
// arbitrary order, so a Stream object can be torn down before the tensors that
// still need it.  Counting here instead makes a handle a plain integer on the
// caller's side, with no finalization order to get wrong.
struct StreamBox {
    om::Stream stream;
    // atomic because ctypes drops the GIL around every call, so two Python
    // threads can retain/release the same stream concurrently.
    std::atomic<long> refs{1};
};

// A stream handle of nullptr means the default (null) stream — the same trick
// om::Stream::default_stream() plays inside the C++ API, so the synchronous and
// asynchronous entry points share one code path.
const om::Stream& om_deref_stream(const void* handle) {
    static const om::Stream s_default = om::Stream::default_stream();
    return handle ? static_cast<const StreamBox*>(handle)->stream : s_default;
}

void om_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Row-major offset for `n` coordinates, bounds-checked against the shape.
size_t om_flat_index(const std::vector<size_t>& shape,
                     const std::vector<size_t>& stride,
                     const size_t* indices, size_t n)
{
    if (n != shape.size())
        throw std::invalid_argument("index: expected " + std::to_string(shape.size()) +
                                    " indices, got " + std::to_string(n));
    size_t flat = 0;
    for (size_t i = 0; i < n; ++i) {
        if (indices[i] >= shape[i])
            throw std::out_of_range("index " + std::to_string(indices[i]) +
                                    " out of range for axis " + std::to_string(i) +
                                    " with size " + std::to_string(shape[i]));
        flat += indices[i] * stride[i];
    }
    return flat;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Exception guards.  Variadic so bodies may contain top-level commas.
// ─────────────────────────────────────────────────────────────────────────────

#define OM_GUARD_PTR(...)                                                       \
    try { __VA_ARGS__ }                                                         \
    catch (const std::exception& e) { set_err(errbuf, errbuf_len, e.what()); return nullptr; } \
    catch (...) { set_err(errbuf, errbuf_len, "OpenMat: unknown error"); return nullptr; }

#define OM_GUARD_INT(...)                                                       \
    try { __VA_ARGS__ }                                                         \
    catch (const std::exception& e) { set_err(errbuf, errbuf_len, e.what()); return -1; } \
    catch (...) { set_err(errbuf, errbuf_len, "OpenMat: unknown error"); return -1; }

#define OM_GUARD_VAL(zero, ...)                                                 \
    try { __VA_ARGS__ }                                                         \
    catch (const std::exception& e) { set_err(errbuf, errbuf_len, e.what()); return zero; } \
    catch (...) { set_err(errbuf, errbuf_len, "OpenMat: unknown error"); return zero; }

// ─────────────────────────────────────────────────────────────────────────────
// Runtime / stream API (dtype-independent)
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

int om_cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

int om_cuda_is_available() {
    return om_cuda_device_count() > 0 ? 1 : 0;
}

int om_device_synchronize(char* errbuf, int errbuf_len) {
    OM_GUARD_INT(
        om_cuda_check(cudaDeviceSynchronize(), "om_device_synchronize");
        return 0;
    )
}

// Creates a stream with a reference count of 1.  Pass nullptr anywhere a
// stream handle is expected to use the default stream instead.
void* om_stream_create(char* errbuf, int errbuf_len) {
    OM_GUARD_PTR( return new StreamBox(); )
}

// Adds a reference.  Every tensor allocated on a stream must hold one for as
// long as it lives, or its deallocation will target a destroyed stream.
void om_stream_retain(void* handle) {
    if (handle)
        static_cast<StreamBox*>(handle)->refs.fetch_add(1, std::memory_order_relaxed);
}

// Drops a reference, destroying the stream when the last one goes.
void om_stream_release(void* handle) {
    if (!handle) return;
    StreamBox* box = static_cast<StreamBox*>(handle);
    if (box->refs.fetch_sub(1, std::memory_order_acq_rel) <= 1)
        delete box;
}

void om_stream_destroy(void* handle) {
    om_stream_release(handle);
}

int om_stream_synchronize(void* handle, char* errbuf, int errbuf_len) {
    OM_GUARD_INT(
        om_deref_stream(handle).synchronize();
        return 0;
    )
}

// Borrowed cudaStream_t behind the handle (nullptr for the default stream) —
// useful for interop with other CUDA libraries.
void* om_stream_handle(void* handle) {
    return static_cast<void*>(om_deref_stream(handle).get());
}

} // extern "C"

// ─────────────────────────────────────────────────────────────────────────────
// Per-dtype tensor APIs
// ─────────────────────────────────────────────────────────────────────────────

#define OM_T   float
#define OM_SFX float
#include "openmat_capi_impl.inc"
#undef OM_T
#undef OM_SFX

#define OM_T   int
#define OM_SFX int
#include "openmat_capi_impl.inc"
#undef OM_T
#undef OM_SFX
