/**
 * Per-dtype body of the OpenMat C-ABI.
 *
 * This file is NOT compiled on its own — openmat_capi.cpp includes it once per
 * supported dtype with OM_T (the C++ element type) and OM_SFX (the token that
 * goes into the exported symbol names) defined.  Including it twice is what
 * gives us om_tensor_float_* and om_tensor_int_* from a single body.
 *
 * Conventions (see openmat_capi.cpp for the shared helpers):
 *   - handles are opaque void* to heap Tensor<OM_T>; one _destroy per _create
 *   - pointer-returning functions return nullptr on failure
 *   - int-returning functions return 0 on success, non-zero on failure
 *   - both write the message into (errbuf, errbuf_len)
 *   - a stream handle of nullptr means the default (null) stream
 */

#if !defined(OM_T) || !defined(OM_SFX)
#  error "openmat_capi_impl.inc requires OM_T and OM_SFX to be defined"
#endif

#define OM_CAT_(a, b) a##b
#define OM_CAT(a, b) OM_CAT_(a, b)
// om_tensor_<OM_SFX>_<name>
#define OM_FN(name) OM_CAT(OM_CAT(om_tensor_, OM_SFX), OM_CAT(_, name))

#define OM_TT om::Tensor<OM_T>

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void* OM_FN(create)(const size_t* shape, size_t rank,
                    int on_cuda,
                    char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(std::vector<size_t>(shape, shape + rank), om_make_device(on_cuda));
    )
}

void* OM_FN(zeros)(const size_t* shape, size_t rank,
                   int on_cuda,
                   char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(OM_TT::zeros(std::vector<size_t>(shape, shape + rank),
                                      om_make_device(on_cuda)));
    )
}

void* OM_FN(ones)(const size_t* shape, size_t rank,
                  int on_cuda,
                  char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(OM_TT::ones(std::vector<size_t>(shape, shape + rank),
                                     om_make_device(on_cuda)));
    )
}

void* OM_FN(full)(const size_t* shape, size_t rank,
                  OM_T value, int on_cuda,
                  char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(OM_TT::full(std::vector<size_t>(shape, shape + rank), value,
                                     om_make_device(on_cuda)));
    )
}

// Copies `n` elements from the host buffer `data` into a new tensor on `device`.
void* OM_FN(from_buffer)(const OM_T* data, size_t n,
                         const size_t* shape, size_t rank,
                         int on_cuda,
                         char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(OM_TT::from_vector(std::vector<OM_T>(data, data + n),
                                            std::vector<size_t>(shape, shape + rank),
                                            om_make_device(on_cuda)));
    )
}

// Same, but the H2D copy is enqueued on `stream` instead of the legacy default
// stream, so it does not serialize against work running on other streams.
//
// It still returns only once the copy has consumed `data`: from_vector stages
// through a temporary std::vector that dies with this call, so the stream is
// synchronized before returning rather than leaving a dangling source pointer.
void* OM_FN(from_buffer_stream)(const OM_T* data, size_t n,
                                const size_t* shape, size_t rank,
                                int on_cuda, void* stream,
                                char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        const om::Stream& s = om_deref_stream(stream);
        OM_TT* t = new OM_TT(OM_TT::from_vector(std::vector<OM_T>(data, data + n),
                                                std::vector<size_t>(shape, shape + rank),
                                                om_make_device(on_cuda), s));
        try { s.synchronize(); } catch (...) { delete t; throw; }
        return t;
    )
}

void OM_FN(destroy)(void* handle)
{
    delete static_cast<OM_TT*>(handle);
}

void* OM_FN(copy)(const void* handle, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(*static_cast<const OM_TT*>(handle));
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata (infallible — no errbuf)
// ─────────────────────────────────────────────────────────────────────────────

size_t OM_FN(rank)(const void* handle) {
    return static_cast<const OM_TT*>(handle)->rank();
}

size_t OM_FN(size)(const void* handle) {
    return static_cast<const OM_TT*>(handle)->size();
}

size_t OM_FN(itemsize)(const void* handle) {
    (void)handle;
    return sizeof(OM_T);
}

// Writes rank() values into `out` (caller must provide the space).
void OM_FN(shape)(const void* handle, size_t* out) {
    const auto sh = static_cast<const OM_TT*>(handle)->shape();
    std::memcpy(out, sh.data(), sh.size() * sizeof(size_t));
}

// Writes rank() values into `out`; strides are in elements, not bytes.
void OM_FN(stride)(const void* handle, size_t* out) {
    const auto st = static_cast<const OM_TT*>(handle)->stride();
    std::memcpy(out, st.data(), st.size() * sizeof(size_t));
}

int OM_FN(on_cuda)(const void* handle) {
    return static_cast<const OM_TT*>(handle)->device_type() == om::DEVICE_TYPE::CUDA ? 1 : 0;
}

int OM_FN(device_id)(const void* handle) {
    return static_cast<const OM_TT*>(handle)->device().m_Id;
}

// Borrowed pointer into a static string — do not free.
const char* OM_FN(dtype)(const void* handle) {
    (void)handle;
    static const std::string s = om::dtype<OM_T>();
    return s.c_str();
}

// Raw pointer to the underlying buffer (device pointer for CUDA tensors).
// Borrowed: valid only while the tensor is alive. Used to build
// __array_interface__ / __cuda_array_interface__ on the Python side.
void* OM_FN(data_ptr)(void* handle) {
    return static_cast<void*>(static_cast<OM_TT*>(handle)->view().data);
}

// ─────────────────────────────────────────────────────────────────────────────
// Data access
// ─────────────────────────────────────────────────────────────────────────────

// Copies every element into the host buffer `out` (cudaMemcpy if on GPU).
int OM_FN(to_host)(const void* handle, OM_T* out, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->copyToHost(out);
        return 0;
    )
}

// Copies a host tensor into the device buffer `dest` (see data_ptr).
int OM_FN(to_device)(const void* handle, void* dest, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->copyToDevice(static_cast<OM_T*>(dest));
        return 0;
    )
}

int OM_FN(fill)(void* handle, OM_T value, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<OM_TT*>(handle)->fill(value);
        return 0;
    )
}

// Single-element read. `indices` holds `n` coordinates (n must equal rank).
int OM_FN(get_item)(const void* handle, const size_t* indices, size_t n,
                    OM_T* out, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        const OM_TT* t = static_cast<const OM_TT*>(handle);
        size_t flat = om_flat_index(t->shape(), t->stride(), indices, n);
        const OM_T* base = t->view().data;
        if (t->device_type() == om::DEVICE_TYPE::CPU) {
            *out = base[flat];
        } else {
            om_cuda_check(cudaMemcpy(out, base + flat, sizeof(OM_T),
                                     cudaMemcpyDeviceToHost), "get_item");
        }
        return 0;
    )
}

// Single-element write, mirroring get_item.
int OM_FN(set_item)(void* handle, const size_t* indices, size_t n,
                    OM_T value, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        OM_TT* t = static_cast<OM_TT*>(handle);
        size_t flat = om_flat_index(t->shape(), t->stride(), indices, n);
        OM_T* base = t->view().data;
        if (t->device_type() == om::DEVICE_TYPE::CPU) {
            base[flat] = value;
        } else {
            om_cuda_check(cudaMemcpy(base + flat, &value, sizeof(OM_T),
                                     cudaMemcpyHostToDevice), "set_item");
        }
        return 0;
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Device transfer
// ─────────────────────────────────────────────────────────────────────────────

void* OM_FN(cpu)(const void* handle, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->cpu()); )
}

void* OM_FN(cuda)(const void* handle, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->cuda()); )
}

void* OM_FN(cpu_stream)(const void* handle, void* stream, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->cpu(om_deref_stream(stream))); )
}

void* OM_FN(cuda_stream)(const void* handle, void* stream, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->cuda(om_deref_stream(stream))); )
}

// `device` is a string such as "cpu", "cuda" or "cuda:0".
void* OM_FN(to)(const void* handle, const char* device, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->to(om::Device(device))); )
}

void* OM_FN(to_stream)(const void* handle, const char* device, void* stream,
                       char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(static_cast<const OM_TT*>(handle)->to(om::Device(device),
                                                               om_deref_stream(stream)));
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic — tensor × tensor, sync and stream forms
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_BINOP_TT(name, method)                                        \
void* OM_FN(name)(const void* lhs, const void* rhs,                             \
                  char* errbuf, int errbuf_len)                                 \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(lhs)->method(                \
            *static_cast<const OM_TT*>(rhs)));                                  \
    )                                                                           \
}                                                                               \
void* OM_FN(name##_stream)(const void* lhs, const void* rhs, void* stream,      \
                           char* errbuf, int errbuf_len)                        \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(lhs)->method(                \
            *static_cast<const OM_TT*>(rhs), om_deref_stream(stream)));         \
    )                                                                           \
}

OM_DEFINE_BINOP_TT(add,    add)
OM_DEFINE_BINOP_TT(sub,    sub)
OM_DEFINE_BINOP_TT(mul,    mul)
OM_DEFINE_BINOP_TT(div,    div)
OM_DEFINE_BINOP_TT(matmul, matmul)

#undef OM_DEFINE_BINOP_TT

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic — tensor × scalar, sync and stream forms
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_BINOP_TS(name, method)                                        \
void* OM_FN(name##_scalar)(const void* handle, OM_T scalar,                     \
                           char* errbuf, int errbuf_len)                        \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(handle)->method(scalar));    \
    )                                                                           \
}                                                                               \
void* OM_FN(name##_scalar_stream)(const void* handle, OM_T scalar, void* stream,\
                                  char* errbuf, int errbuf_len)                 \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(handle)->method(             \
            scalar, om_deref_stream(stream)));                                  \
    )                                                                           \
}

OM_DEFINE_BINOP_TS(add, add)
OM_DEFINE_BINOP_TS(sub, sub)
OM_DEFINE_BINOP_TS(mul, mul)
OM_DEFINE_BINOP_TS(div, div)

#undef OM_DEFINE_BINOP_TS

// ─────────────────────────────────────────────────────────────────────────────
// In-place arithmetic and destination-provided ops
//
// These return int, not a handle: nothing is allocated and no ownership
// changes — the result lands in a tensor the caller already owns, and every
// handle involved stays valid. That is the point of them. A loop that would
// otherwise create and destroy one tensor per op per iteration keeps a single
// buffer for its whole life.
//
// `_out` takes the destination handle last. It may be one of the operands for
// the elementwise family (that is what the in-place form is); matmul_out,
// transpose_out and permute_out reject an aliased destination on the C++ side.
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_INPLACE_TT(name, method)                                      \
int OM_FN(name##_inplace)(void* lhs, const void* rhs,                           \
                          char* errbuf, int errbuf_len)                         \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<OM_TT*>(lhs)->method##_(*static_cast<const OM_TT*>(rhs));   \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_inplace_stream)(void* lhs, const void* rhs, void* stream,      \
                                 char* errbuf, int errbuf_len)                  \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<OM_TT*>(lhs)->method##_(*static_cast<const OM_TT*>(rhs),    \
                                            om_deref_stream(stream));           \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_out)(const void* lhs, const void* rhs, void* out,              \
                      char* errbuf, int errbuf_len)                             \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(lhs)->method##_out(                           \
            *static_cast<const OM_TT*>(rhs), *static_cast<OM_TT*>(out));        \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_out_stream)(const void* lhs, const void* rhs, void* out,       \
                             void* stream, char* errbuf, int errbuf_len)        \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(lhs)->method##_out(                           \
            *static_cast<const OM_TT*>(rhs), *static_cast<OM_TT*>(out),         \
            om_deref_stream(stream));                                           \
        return 0;                                                               \
    )                                                                           \
}

OM_DEFINE_INPLACE_TT(add, add)
OM_DEFINE_INPLACE_TT(sub, sub)
OM_DEFINE_INPLACE_TT(mul, mul)
OM_DEFINE_INPLACE_TT(div, div)

#undef OM_DEFINE_INPLACE_TT

// matmul has no in-place form: its kernel reads elements it does not write.
int OM_FN(matmul_out)(const void* lhs, const void* rhs, void* out,
                      char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(lhs)->matmul_out(*static_cast<const OM_TT*>(rhs),
                                                   *static_cast<OM_TT*>(out));
        return 0;
    )
}

int OM_FN(matmul_out_stream)(const void* lhs, const void* rhs, void* out, void* stream,
                             char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(lhs)->matmul_out(*static_cast<const OM_TT*>(rhs),
                                                   *static_cast<OM_TT*>(out),
                                                   om_deref_stream(stream));
        return 0;
    )
}

#define OM_DEFINE_INPLACE_TS(name, method)                                      \
int OM_FN(name##_scalar_inplace)(void* handle, OM_T scalar,                     \
                                 char* errbuf, int errbuf_len)                  \
{                                                                               \
    OM_GUARD_INT( static_cast<OM_TT*>(handle)->method##_(scalar); return 0; )    \
}                                                                               \
int OM_FN(name##_scalar_inplace_stream)(void* handle, OM_T scalar, void* stream, \
                                        char* errbuf, int errbuf_len)           \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<OM_TT*>(handle)->method##_(scalar, om_deref_stream(stream)); \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_scalar_out)(const void* handle, OM_T scalar, void* out,        \
                             char* errbuf, int errbuf_len)                      \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(handle)->method##_out(scalar,                 \
                                                        *static_cast<OM_TT*>(out)); \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_scalar_out_stream)(const void* handle, OM_T scalar, void* out, \
                                    void* stream, char* errbuf, int errbuf_len) \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(handle)->method##_out(scalar,                 \
                                                        *static_cast<OM_TT*>(out), \
                                                        om_deref_stream(stream)); \
        return 0;                                                               \
    )                                                                           \
}

OM_DEFINE_INPLACE_TS(add, add)
OM_DEFINE_INPLACE_TS(sub, sub)
OM_DEFINE_INPLACE_TS(mul, mul)
OM_DEFINE_INPLACE_TS(div, div)

#undef OM_DEFINE_INPLACE_TS

#define OM_DEFINE_INPLACE_UNARY(name)                                           \
int OM_FN(name##_inplace)(void* handle, char* errbuf, int errbuf_len)           \
{                                                                               \
    OM_GUARD_INT( static_cast<OM_TT*>(handle)->name##_(); return 0; )            \
}                                                                               \
int OM_FN(name##_inplace_stream)(void* handle, void* stream,                    \
                                 char* errbuf, int errbuf_len)                  \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<OM_TT*>(handle)->name##_(om_deref_stream(stream));          \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_out)(const void* handle, void* out, char* errbuf, int errbuf_len) \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(handle)->name##_out(*static_cast<OM_TT*>(out)); \
        return 0;                                                               \
    )                                                                           \
}                                                                               \
int OM_FN(name##_out_stream)(const void* handle, void* out, void* stream,       \
                             char* errbuf, int errbuf_len)                      \
{                                                                               \
    OM_GUARD_INT(                                                               \
        static_cast<const OM_TT*>(handle)->name##_out(*static_cast<OM_TT*>(out), \
                                                      om_deref_stream(stream)); \
        return 0;                                                               \
    )                                                                           \
}

OM_DEFINE_INPLACE_UNARY(relu)
OM_DEFINE_INPLACE_UNARY(sigmoid)

#undef OM_DEFINE_INPLACE_UNARY

int OM_FN(transpose_out)(const void* handle, void* out, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->transpose_out(*static_cast<OM_TT*>(out));
        return 0;
    )
}

int OM_FN(transpose_out_stream)(const void* handle, void* out, void* stream,
                                char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->transpose_out(*static_cast<OM_TT*>(out),
                                                         om_deref_stream(stream));
        return 0;
    )
}

int OM_FN(permute_out)(const void* handle, const size_t* axes, size_t n, void* out,
                       char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->permute_out(
            std::vector<size_t>(axes, axes + n), *static_cast<OM_TT*>(out));
        return 0;
    )
}

int OM_FN(permute_out_stream)(const void* handle, const size_t* axes, size_t n, void* out,
                              void* stream, char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<const OM_TT*>(handle)->permute_out(
            std::vector<size_t>(axes, axes + n), *static_cast<OM_TT*>(out),
            om_deref_stream(stream));
        return 0;
    )
}

// fill() has always existed; this is the same thing with a stream.
int OM_FN(fill_stream)(void* handle, OM_T value, void* stream,
                       char* errbuf, int errbuf_len)
{
    OM_GUARD_INT(
        static_cast<OM_TT*>(handle)->fill_(value, om_deref_stream(stream));
        return 0;
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Reductions — synchronous, return a host scalar
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_REDUCTION(name, method)                                       \
OM_T OM_FN(name)(const void* handle, char* errbuf, int errbuf_len)              \
{                                                                               \
    OM_GUARD_VAL(static_cast<OM_T>(0),                                          \
        return static_cast<const OM_TT*>(handle)->method();                     \
    )                                                                           \
}

OM_DEFINE_REDUCTION(sum,  sum)
OM_DEFINE_REDUCTION(mean, mean)
OM_DEFINE_REDUCTION(min,  min)
OM_DEFINE_REDUCTION(max,  max)

#undef OM_DEFINE_REDUCTION

// ─────────────────────────────────────────────────────────────────────────────
// Shape manipulation
// ─────────────────────────────────────────────────────────────────────────────

void* OM_FN(reshape)(const void* handle, const size_t* new_shape, size_t new_rank,
                     char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(static_cast<const OM_TT*>(handle)->reshape(
            std::vector<size_t>(new_shape, new_shape + new_rank)));
    )
}

void* OM_FN(flatten)(const void* handle, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->flatten()); )
}

void* OM_FN(squeeze)(const void* handle, size_t axis, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->squeeze(axis)); )
}

void* OM_FN(unsqueeze)(const void* handle, size_t axis, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->unsqueeze(axis)); )
}

void* OM_FN(transpose)(const void* handle, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->transpose()); )
}

void* OM_FN(transpose_stream)(const void* handle, void* stream,
                              char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(static_cast<const OM_TT*>(handle)->transpose(om_deref_stream(stream)));
    )
}

void* OM_FN(permute)(const void* handle, const size_t* axes, size_t n,
                     char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(static_cast<const OM_TT*>(handle)->permute(
            std::vector<size_t>(axes, axes + n)));
    )
}

void* OM_FN(permute_stream)(const void* handle, const size_t* axes, size_t n,
                            void* stream, char* errbuf, int errbuf_len)
{
    OM_GUARD_PTR(
        return new OM_TT(static_cast<const OM_TT*>(handle)->permute(
            std::vector<size_t>(axes, axes + n), om_deref_stream(stream)));
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused ops — unary
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_UNARY_FUSED(name)                                             \
void* OM_FN(name)(const void* handle, char* errbuf, int errbuf_len)             \
{                                                                               \
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->name()); ) \
}                                                                               \
void* OM_FN(name##_stream)(const void* handle, void* stream,                    \
                           char* errbuf, int errbuf_len)                        \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(handle)->name(               \
            om_deref_stream(stream)));                                          \
    )                                                                           \
}

OM_DEFINE_UNARY_FUSED(relu)
OM_DEFINE_UNARY_FUSED(sigmoid)

#undef OM_DEFINE_UNARY_FUSED

// scale_shift(x) = x * scale + shift ; shift_scale(x) = (x + shift) * scale
#define OM_DEFINE_AFFINE(name)                                                  \
void* OM_FN(name)(const void* handle, OM_T a, OM_T b,                           \
                  char* errbuf, int errbuf_len)                                 \
{                                                                               \
    OM_GUARD_PTR( return new OM_TT(static_cast<const OM_TT*>(handle)->name(a, b)); ) \
}

OM_DEFINE_AFFINE(scale_shift)
OM_DEFINE_AFFINE(shift_scale)

#undef OM_DEFINE_AFFINE

// ─────────────────────────────────────────────────────────────────────────────
// Fused ops — binary-then-scalar
// ─────────────────────────────────────────────────────────────────────────────

#define OM_DEFINE_BINARY_FUSED(name)                                            \
void* OM_FN(name)(const void* lhs, const void* rhs, OM_T scalar,                \
                  char* errbuf, int errbuf_len)                                 \
{                                                                               \
    OM_GUARD_PTR(                                                               \
        return new OM_TT(static_cast<const OM_TT*>(lhs)->name(                  \
            *static_cast<const OM_TT*>(rhs), scalar));                          \
    )                                                                           \
}

OM_DEFINE_BINARY_FUSED(fused_add_mul)
OM_DEFINE_BINARY_FUSED(fused_sub_mul)
OM_DEFINE_BINARY_FUSED(fused_mul_add)
OM_DEFINE_BINARY_FUSED(fused_div_add)

#undef OM_DEFINE_BINARY_FUSED

} // extern "C"

#undef OM_TT
#undef OM_FN
#undef OM_CAT
#undef OM_CAT_
