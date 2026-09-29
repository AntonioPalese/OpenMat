// DLPack ownership, checked at the C-ABI boundary.
//
// python/tests/test_dlpack.py covers interop with NumPy and PyTorch; this
// suite pins the part a Python test cannot observe: that every export's
// deleter frees its storage and releases its stream reference, and that an
// import runs the producer's deleter exactly once — when the last view dies,
// never for an import that was refused. Run under memcheck, a missed free or
// a double free shows up here rather than as a leak in someone else's
// process.
#include "test_helpers.h"
#include "dlpack.h"

extern "C" {
void* om_stream_create(char*, int);
void  om_stream_release(void*);
void* om_tensor_float_create(const size_t*, size_t, int, char*, int);
void* om_tensor_float_add_stream(const void*, const void*, void*, char*, int);
void* om_tensor_float_slice(const void*, size_t, size_t, size_t, size_t, char*, int);
void  om_tensor_float_destroy(void*);
int   om_tensor_float_fill(void*, float, char*, int);
void* om_tensor_float_to_dlpack(const void*, void*, char*, int);
void* om_tensor_float_from_dlpack(void*, char*, int);
void* om_tensor_int_from_dlpack(void*, char*, int);
void  om_dlpack_delete(void*);
}

namespace {

char g_err[512];

// A producer-side tensor over `data`, whose deleter counts its calls.
struct Producer {
    std::vector<float> host;
    int64_t shape[2];
    int64_t strides[2];
    int deleted = 0;
    DLManagedTensor managed{};

    Producer(size_t rows, size_t cols) : host(rows * cols, 1.0f)
    {
        shape[0] = static_cast<int64_t>(rows);
        shape[1] = static_cast<int64_t>(cols);
        strides[0] = static_cast<int64_t>(cols);
        strides[1] = 1;
        DLTensor& t = managed.dl_tensor;
        t.data = host.data();
        t.device = {kDLCPU, 0};
        t.ndim = 2;
        t.dtype = {kDLFloat, 32, 1};
        t.shape = shape;
        t.strides = strides;
        t.byte_offset = 0;
        managed.manager_ctx = this;
        managed.deleter = [](DLManagedTensor* self) {
            ++static_cast<Producer*>(self->manager_ctx)->deleted;
        };
    }
};

} // namespace

TEST(DLPack, ImportRunsProducerDeleterOnceAfterLastView)
{
    Producer p(4, 6);
    void* t = om_tensor_float_from_dlpack(&p.managed, g_err, sizeof g_err);
    ASSERT_NE(t, nullptr) << g_err;
    void* view = om_tensor_float_slice(t, 1, 1, 6, 2, g_err, sizeof g_err);
    ASSERT_NE(view, nullptr) << g_err;

    // Writes land in the producer's memory: no copy was made.
    ASSERT_EQ(om_tensor_float_fill(view, 7.0f, g_err, sizeof g_err), 0) << g_err;
    EXPECT_EQ(p.host[1], 7.0f);
    EXPECT_EQ(p.host[0], 1.0f);

    om_tensor_float_destroy(t);
    EXPECT_EQ(p.deleted, 0);                  // the view still holds the storage
    om_tensor_float_destroy(view);
    EXPECT_EQ(p.deleted, 1);
}

TEST(DLPack, RefusedImportLeavesTheTensorWithItsOwner)
{
    Producer p(2, 2);
    // Wrong dtype for the int entry point.
    EXPECT_EQ(om_tensor_int_from_dlpack(&p.managed, g_err, sizeof g_err), nullptr);
    EXPECT_NE(std::string(g_err).find("dtype"), std::string::npos);
    // Negative strides.
    p.strides[1] = -1;
    EXPECT_EQ(om_tensor_float_from_dlpack(&p.managed, g_err, sizeof g_err), nullptr);
    EXPECT_EQ(p.deleted, 0);
    // The owner still has to release it exactly once.
    om_dlpack_delete(&p.managed);
    EXPECT_EQ(p.deleted, 1);
}

TEST(DLPack, ExportDescribesTheViewAndOutlivesTheHandle)
{
    const size_t shape[2] = {4, 6};
    void* base = om_tensor_float_create(shape, 2, 0, g_err, sizeof g_err);
    ASSERT_NE(base, nullptr) << g_err;
    ASSERT_EQ(om_tensor_float_fill(base, 3.0f, g_err, sizeof g_err), 0);
    void* view = om_tensor_float_slice(base, 1, 1, 6, 2, g_err, sizeof g_err);

    auto* m = static_cast<DLManagedTensor*>(om_tensor_float_to_dlpack(view, nullptr, g_err, sizeof g_err));
    ASSERT_NE(m, nullptr) << g_err;
    om_tensor_float_destroy(view);
    om_tensor_float_destroy(base);            // every OpenMat handle is gone

    const DLTensor& t = m->dl_tensor;
    EXPECT_EQ(t.device.device_type, kDLCPU);
    EXPECT_EQ(t.ndim, 2);
    EXPECT_EQ(t.dtype.code, kDLFloat);
    EXPECT_EQ(t.shape[0], 4);
    EXPECT_EQ(t.shape[1], 3);
    EXPECT_EQ(t.strides[0], 6);
    EXPECT_EQ(t.strides[1], 2);
    EXPECT_EQ(static_cast<float*>(t.data)[0], 3.0f);   // still valid memory
    m->deleter(m);
}

// The export holds a reference on the stream its storage frees on, so the
// consumer can release it after the stream's own handle is gone. memcheck in
// the gpu CI job is what would catch a free on a destroyed stream.
TEST(DLPack, ExportKeepsItsStreamAlive)
{
    OM_REQUIRE_CUDA();
    const size_t shape[1] = {1 << 16};
    void* stream = om_stream_create(g_err, sizeof g_err);
    ASSERT_NE(stream, nullptr) << g_err;
    void* a = om_tensor_float_create(shape, 1, 1, g_err, sizeof g_err);
    ASSERT_EQ(om_tensor_float_fill(a, 1.0f, g_err, sizeof g_err), 0);
    void* r = om_tensor_float_add_stream(a, a, stream, g_err, sizeof g_err);
    ASSERT_NE(r, nullptr) << g_err;

    auto* m = static_cast<DLManagedTensor*>(om_tensor_float_to_dlpack(r, stream, g_err, sizeof g_err));
    ASSERT_NE(m, nullptr) << g_err;
    EXPECT_EQ(m->dl_tensor.device.device_type, kDLCUDA);

    om_tensor_float_destroy(r);
    om_tensor_float_destroy(a);
    om_stream_release(stream);                // the caller's reference

    float first = 0;
    cudaDeviceSynchronize();
    cudaMemcpy(&first, m->dl_tensor.data, sizeof first, cudaMemcpyDeviceToHost);
    EXPECT_EQ(first, 2.0f);
    m->deleter(m);                            // frees the storage, then the stream
    cudaDeviceSynchronize();
}
