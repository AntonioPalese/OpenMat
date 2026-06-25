#include "test_helpers.h"
#include "allocator.h"
#include "stream.h"
#include "device_tensor_view.cuh"

// ── DeviceTensorView inline metadata ─────────────────────────────────────────

TEST(DeviceTensorViewInline, ConstructionNoCudaMalloc) {
    // Shape/stride live as inline arrays — no device allocation happens here.
    size_t shape[]  = {3, 4};
    size_t stride[] = {4, 1};
    float dummy     = 0.f;

    om::DeviceTensorView<float> dtv(&dummy, shape, stride, 2);

    EXPECT_EQ(dtv.rank,      2u);
    EXPECT_EQ(dtv.shape[0],  3u);
    EXPECT_EQ(dtv.shape[1],  4u);
    EXPECT_EQ(dtv.stride[0], 4u);
    EXPECT_EQ(dtv.stride[1], 1u);
}

TEST(DeviceTensorViewInline, CopyableByValue) {
    size_t shape[]  = {2, 3};
    size_t stride[] = {3, 1};
    float dummy = 0.f;

    om::DeviceTensorView<float> a(&dummy, shape, stride, 2);
    om::DeviceTensorView<float> b = a;  // should compile and copy inline arrays

    EXPECT_EQ(b.rank,      2u);
    EXPECT_EQ(b.shape[0],  2u);
    EXPECT_EQ(b.shape[1],  3u);
}

// ── GpuAllocator async methods ────────────────────────────────────────────────

TEST(GpuAllocatorAsync, AllocateAndFreeAsync) {
    om::Stream s;
    om::GpuAllocator<float> alloc;

    float* ptr = alloc.allocate_async(64, s.get());
    EXPECT_NE(ptr, nullptr);

    alloc.deallocate_async(ptr, s.get());
    s.synchronize();  // ensure free completes
}

TEST(GpuAllocatorAsync, CopyD2DAsync) {
    om::Stream s;
    om::GpuAllocator<float> alloc;

    // Allocate source on GPU and fill it from host
    float* src = alloc.allocate(4);
    float h_src[4] = {1.f, 2.f, 3.f, 4.f};
    CUDA_CALL(cudaMemcpy(src, h_src, 4 * sizeof(float), cudaMemcpyHostToDevice));

    // Async D2D copy to dst
    float* dst = alloc.allocate_async(4, s.get());
    alloc.copy_async(dst, src, 4, s.get());
    s.synchronize();

    // Read back
    float h_dst[4] = {};
    CUDA_CALL(cudaMemcpy(h_dst, dst, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(h_dst[i], h_src[i]);

    alloc.deallocate(src);
    alloc.deallocate_async(dst, s.get());
    s.synchronize();
}

TEST(GpuAllocatorAsync, CopyH2DAsync) {
    om::Stream s;
    om::GpuAllocator<float> alloc;

    float h[4] = {10.f, 20.f, 30.f, 40.f};
    float* d = alloc.allocate_async(4, s.get());

    alloc.copy_host_to_device_async(d, h, 4, s.get());
    s.synchronize();

    float h_back[4] = {};
    CUDA_CALL(cudaMemcpy(h_back, d, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(h_back[i], h[i]);

    alloc.deallocate_async(d, s.get());
    s.synchronize();
}

TEST(GpuAllocatorAsync, CopyD2HAsync) {
    om::Stream s;
    om::GpuAllocator<float> alloc;

    float h_src[4] = {5.f, 6.f, 7.f, 8.f};
    float* d = alloc.allocate(4);
    CUDA_CALL(cudaMemcpy(d, h_src, 4 * sizeof(float), cudaMemcpyHostToDevice));

    float h_dst[4] = {};
    alloc.copy_device_to_host_async(h_dst, d, 4, s.get());
    s.synchronize();

    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(h_dst[i], h_src[i]);

    alloc.deallocate(d);
}

// ── Tensor::to(device, stream) ────────────────────────────────────────────────

TEST(TensorToAsync, CPU_to_GPU) {
    auto cpu_t = Tensor<float>::from_vector({1.f,2.f,3.f,4.f}, {4}, Device("cpu:0"));

    om::Stream s;
    auto gpu_t = cpu_t.to(Device("cuda:0"), s);
    s.synchronize();

    auto v = to_host(gpu_t);
    ASSERT_EQ(v.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i + 1));
}

TEST(TensorToAsync, GPU_to_CPU) {
    auto gpu_t = Tensor<float>::from_vector({10.f,20.f,30.f}, {3}, Device("cuda:0"));

    om::Stream s;
    auto cpu_t = gpu_t.to(Device("cpu:0"), s);
    s.synchronize();

    std::vector<float> v(3);
    cpu_t.copyToHost(v.data());
    EXPECT_FLOAT_EQ(v[0], 10.f);
    EXPECT_FLOAT_EQ(v[1], 20.f);
    EXPECT_FLOAT_EQ(v[2], 30.f);
}

TEST(TensorToAsync, CpuShorthand) {
    auto gpu_t = Tensor<float>::from_vector({1.f, 2.f}, {2}, Device("cuda:0"));
    om::Stream s;
    auto cpu_t = gpu_t.cpu(s);
    s.synchronize();

    std::vector<float> v(2);
    cpu_t.copyToHost(v.data());
    EXPECT_FLOAT_EQ(v[0], 1.f);
    EXPECT_FLOAT_EQ(v[1], 2.f);
}

TEST(TensorToAsync, CudaShorthand) {
    auto cpu_t = Tensor<float>::from_vector({7.f, 8.f}, {2}, Device("cpu:0"));
    om::Stream s;
    auto gpu_t = cpu_t.cuda(s);
    s.synchronize();

    auto v = to_host(gpu_t);
    EXPECT_FLOAT_EQ(v[0], 7.f);
    EXPECT_FLOAT_EQ(v[1], 8.f);
}

// ── Tensor::from_vector with stream ──────────────────────────────────────────

TEST(TensorFromVectorAsync, GPU_async) {
    om::Stream s;
    auto t = Tensor<float>::from_vector({3.f,1.f,4.f,1.f,5.f}, {5}, Device("cuda:0"), s);
    s.synchronize();

    auto v = to_host(t);
    ASSERT_EQ(v.size(), 5u);
    EXPECT_FLOAT_EQ(v[0], 3.f);
    EXPECT_FLOAT_EQ(v[2], 4.f);
    EXPECT_FLOAT_EQ(v[4], 5.f);
}

TEST(TensorFromVectorAsync, ResultMatchesSyncVersion) {
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<size_t> sh  = {2, 3};

    auto sync_t  = Tensor<float>::from_vector(data, sh, Device("cuda:0"));
    om::Stream s;
    auto async_t = Tensor<float>::from_vector(data, sh, Device("cuda:0"), s);
    s.synchronize();

    auto vs = to_host(sync_t);
    auto va = to_host(async_t);
    ASSERT_EQ(vs.size(), va.size());
    for (size_t i = 0; i < vs.size(); ++i)
        EXPECT_FLOAT_EQ(vs[i], va[i]);
}
