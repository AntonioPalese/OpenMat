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
 * returning functions return nullptr on failure.
 */

#include "tensor.cuh"
#include "mat_utils.h"
#include <cstring>
#include <cstdio>

// Convenience: fill errbuf (if non-null, capacity errbuf_len) from exception.
static void set_err(char* errbuf, int errbuf_len, const char* msg) {
    if (errbuf && errbuf_len > 0)
        std::snprintf(errbuf, errbuf_len, "%s", msg);
}

using TF = om::Tensor<float>;
using TI = om::Tensor<int>;

// ─────────────────────────────────────────────────────────────────────────────
// Float tensor API
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

// --- lifecycle ---------------------------------------------------------------

void* om_tensor_float_create(const size_t* shape, size_t rank,
                              int on_cuda,
                              char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(shape, shape + rank);
        om::Device dv(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
        return new TF(sh, dv);
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_zeros(const size_t* shape, size_t rank,
                             int on_cuda,
                             char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(shape, shape + rank);
        om::Device dv(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
        return new TF(TF::zeros(sh, dv));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_ones(const size_t* shape, size_t rank,
                            int on_cuda,
                            char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(shape, shape + rank);
        om::Device dv(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
        return new TF(TF::ones(sh, dv));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_full(const size_t* shape, size_t rank,
                            float value, int on_cuda,
                            char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(shape, shape + rank);
        om::Device dv(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
        return new TF(TF::full(sh, value, dv));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

// from_buffer: copies `n` floats from host buffer `data` into tensor on `device`.
void* om_tensor_float_from_buffer(const float* data, size_t n,
                                   const size_t* shape, size_t rank,
                                   int on_cuda,
                                   char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(shape, shape + rank);
        std::vector<float> v(data, data + n);
        om::Device dv(0, on_cuda ? om::DEVICE_TYPE::CUDA : om::DEVICE_TYPE::CPU);
        return new TF(TF::from_vector(v, sh, dv));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void om_tensor_float_destroy(void* handle)
{
    delete static_cast<TF*>(handle);
}

void* om_tensor_float_copy(const void* handle,
                            char* errbuf, int errbuf_len)
{
    try {
        return new TF(*static_cast<const TF*>(handle));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

// --- metadata ----------------------------------------------------------------

size_t om_tensor_float_rank(const void* handle) {
    return static_cast<const TF*>(handle)->rank();
}

size_t om_tensor_float_size(const void* handle) {
    return static_cast<const TF*>(handle)->size();
}

// Writes rank() values into `out` (caller must provide enough space).
void om_tensor_float_shape(const void* handle, size_t* out) {
    const auto& sh = static_cast<const TF*>(handle)->shape();
    std::memcpy(out, sh.data(), sh.size() * sizeof(size_t));
}

int om_tensor_float_on_cuda(const void* handle) {
    return static_cast<const TF*>(handle)->device_type() == om::DEVICE_TYPE::CUDA ? 1 : 0;
}

// --- data access -------------------------------------------------------------

// Copies all elements to a host buffer (cudaMemcpy if on GPU).
int om_tensor_float_to_host(const void* handle, float* out,
                             char* errbuf, int errbuf_len)
{
    try {
        static_cast<const TF*>(handle)->copyToHost(out);
        return 0;
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return -1;
    }
}

void om_tensor_float_fill(void* handle, float value) {
    static_cast<TF*>(handle)->fill(value);
}

// --- device transfer ---------------------------------------------------------

void* om_tensor_float_cpu(const void* handle,
                           char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->cpu());
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_cuda(const void* handle,
                            char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->cuda());
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

// --- arithmetic (tensor × tensor) -------------------------------------------

#define DEFINE_BINOP_TT(name, method) \
void* om_tensor_float_##name(const void* lhs, const void* rhs, \
                              char* errbuf, int errbuf_len) \
{ \
    try { \
        return new TF(static_cast<const TF*>(lhs)->method(*static_cast<const TF*>(rhs))); \
    } catch (const std::exception& e) { \
        set_err(errbuf, errbuf_len, e.what()); \
        return nullptr; \
    } \
}

DEFINE_BINOP_TT(add,  add)
DEFINE_BINOP_TT(sub,  sub)
DEFINE_BINOP_TT(mul,  mul)
DEFINE_BINOP_TT(div,  div)
DEFINE_BINOP_TT(matmul, matmul)

// --- arithmetic (tensor × scalar) -------------------------------------------

#define DEFINE_BINOP_TS(name, method) \
void* om_tensor_float_##name##_scalar(const void* handle, float scalar, \
                                       char* errbuf, int errbuf_len) \
{ \
    try { \
        return new TF(static_cast<const TF*>(handle)->method(scalar)); \
    } catch (const std::exception& e) { \
        set_err(errbuf, errbuf_len, e.what()); \
        return nullptr; \
    } \
}

DEFINE_BINOP_TS(add, add)
DEFINE_BINOP_TS(sub, sub)
DEFINE_BINOP_TS(mul, mul)
DEFINE_BINOP_TS(div, div)

// --- reductions --------------------------------------------------------------

#define DEFINE_REDUCTION(name, method) \
float om_tensor_float_##name(const void* handle, \
                              char* errbuf, int errbuf_len) \
{ \
    try { \
        return static_cast<const TF*>(handle)->method(); \
    } catch (const std::exception& e) { \
        set_err(errbuf, errbuf_len, e.what()); \
        return 0.0f; \
    } \
}

DEFINE_REDUCTION(sum,  sum)
DEFINE_REDUCTION(mean, mean)
DEFINE_REDUCTION(min,  min)
DEFINE_REDUCTION(max,  max)

// --- shape manipulation ------------------------------------------------------

void* om_tensor_float_reshape(const void* handle,
                               const size_t* new_shape, size_t new_rank,
                               char* errbuf, int errbuf_len)
{
    try {
        std::vector<size_t> sh(new_shape, new_shape + new_rank);
        return new TF(static_cast<const TF*>(handle)->reshape(sh));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_flatten(const void* handle,
                               char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->flatten());
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_squeeze(const void* handle, size_t axis,
                               char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->squeeze(axis));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_unsqueeze(const void* handle, size_t axis,
                                 char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->unsqueeze(axis));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

// --- fused ops ---------------------------------------------------------------

void* om_tensor_float_scale_shift(const void* handle,
                                   float scale, float shift,
                                   char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(handle)->scale_shift(scale, shift));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_fused_add_mul(const void* lhs, const void* rhs,
                                     float scale,
                                     char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(lhs)->fused_add_mul(
            *static_cast<const TF*>(rhs), scale));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

void* om_tensor_float_fused_mul_add(const void* lhs, const void* rhs,
                                     float shift,
                                     char* errbuf, int errbuf_len)
{
    try {
        return new TF(static_cast<const TF*>(lhs)->fused_mul_add(
            *static_cast<const TF*>(rhs), shift));
    } catch (const std::exception& e) {
        set_err(errbuf, errbuf_len, e.what());
        return nullptr;
    }
}

} // extern "C"
