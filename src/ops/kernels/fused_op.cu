#include <stdexcept>
#include "ops/kernels/fused_op.cuh"
#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

namespace om {

    
    template <typename T, typename Op>
    __global__ void apply_op_rank1(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < dst.shape[0])
            dst(x) = op(src(x));
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_rank2(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (y < dst.shape[0] && x < dst.shape[1])
            dst(y, x) = op(src(y, x));
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_rank3(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (z < dst.shape[0] && y < dst.shape[1] && x < dst.shape[2])
            dst(z, y, x) = op(src(z, y, x));
    }

    template <typename T, typename Op>
    __global__ void apply_op_rank4(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t w = threadIdx.x + blockIdx.x * blockDim.x;
        size_t h = threadIdx.y + blockIdx.y * blockDim.y;
        size_t n = blockIdx.z; // N dimension
    
        for (size_t c = threadIdx.z; c < dst.shape[1]; c += blockDim.z) {
            if (n < dst.shape[0] &&
                c < dst.shape[1] &&
                h < dst.shape[2] &&
                w < dst.shape[3]) {
                dst(n, c, h, w) = op(src(n, c, h, w));
            }
        }
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_nd(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elements = dst.size();
    
        if (idx >= total_elements) return;
    
        size_t offset = 0;
        size_t tmp = idx;
        for (size_t d = 0; d < dst.rank; ++d) {
            size_t coord = tmp % dst.shape[d];
            offset += coord * dst.stride[d];
            tmp /= dst.shape[d];
        }
    
        dst[offset] = op(src[offset]);
    }

    template <typename T, typename Op>
    void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op, cudaStream_t stream)
    {
        if (!src.match(dst))
            throw std::invalid_argument("Source and destination must have the same shape");

        switch (dst.rank)
        {
        case 1:
            {
                dim3 threads(16);
                dim3 blocks((dst.shape[0] + 15) / 16);
                apply_op_rank1<<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 2:
            {
                dim3 threads(16, 16);
                dim3 blocks((dst.shape[1] + 15) / 16, (dst.shape[0] + 15) / 16);
                apply_op_rank2<<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 3:
            {
                dim3 threads(8, 8, 8);
                dim3 blocks((dst.shape[2] + 7) / 8, (dst.shape[1] + 7) / 8, (dst.shape[0] + 7) / 8);
                apply_op_rank3<<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 4:
            {
                dim3 threads(8, 8, dst.shape[1] < 8 ? dst.shape[1] : 8);
                dim3 blocks(
                    (dst.shape[3] + threads.x - 1) / threads.x,
                    (dst.shape[2] + threads.y - 1) / threads.y,
                    dst.shape[0]
                );
                apply_op_rank4<<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        default:
            {
                size_t total_elements = dst.size();
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                apply_op_nd<<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        }
        CUDA_CHECK;
        if (stream == nullptr) cudaDeviceSynchronize();
    }

    // ---------------------------------------------------------------------------
    // Binary fused kernels
    // ---------------------------------------------------------------------------

    template <typename T, typename Op>
    __global__ void apply_binary_op_rank1(
        const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs,
        DeviceTensorView<T> dst, Op op)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < dst.shape[0])
            dst(x) = op(lhs(x), rhs(x));
    }

    template <typename T, typename Op>
    __global__ void apply_binary_op_rank2(
        const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs,
        DeviceTensorView<T> dst, Op op)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < dst.shape[0] && x < dst.shape[1])
            dst(y, x) = op(lhs(y, x), rhs(y, x));
    }

    template <typename T, typename Op>
    __global__ void apply_binary_op_rank3(
        const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs,
        DeviceTensorView<T> dst, Op op)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        if (z < dst.shape[0] && y < dst.shape[1] && x < dst.shape[2])
            dst(z, y, x) = op(lhs(z, y, x), rhs(z, y, x));
    }

    template <typename T, typename Op>
    __global__ void apply_binary_op_rank4(
        const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs,
        DeviceTensorView<T> dst, Op op)
    {
        size_t w = threadIdx.x + blockIdx.x * blockDim.x;
        size_t h = threadIdx.y + blockIdx.y * blockDim.y;
        size_t n = blockIdx.z;

        for (size_t c = threadIdx.z; c < dst.shape[1]; c += blockDim.z) {
            if (n < dst.shape[0] && c < dst.shape[1] &&
                h < dst.shape[2] && w < dst.shape[3])
                dst(n, c, h, w) = op(lhs(n, c, h, w), rhs(n, c, h, w));
        }
    }

    template <typename T, typename Op>
    __global__ void apply_binary_op_nd(
        const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs,
        DeviceTensorView<T> dst, Op op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = dst.size();
        if (idx >= total) return;

        size_t offset = 0;
        size_t tmp = idx;
        for (size_t d = 0; d < dst.rank; ++d) {
            size_t coord = tmp % dst.shape[d];
            offset += coord * dst.stride[d];
            tmp /= dst.shape[d];
        }
        dst[offset] = op(lhs[offset], rhs[offset]);
    }

    template <typename T, typename Op>
    void launch_apply_binary_op(const TensorView<const T> lhs, const TensorView<const T> rhs,
                                TensorView<T> dst, Op op, cudaStream_t stream)
    {
        if (!lhs.match(dst) || !rhs.match(dst))
            throw std::invalid_argument("launch_apply_binary_op: all tensors must have the same shape");

        switch (dst.rank)
        {
        case 1:
            {
                dim3 threads(256);
                dim3 blocks((dst.shape[0] + 255) / 256);
                apply_binary_op_rank1<<<blocks, threads, 0, stream>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 2:
            {
                dim3 threads(16, 16);
                dim3 blocks((dst.shape[1] + 15) / 16, (dst.shape[0] + 15) / 16);
                apply_binary_op_rank2<<<blocks, threads, 0, stream>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 3:
            {
                dim3 threads(8, 8, 8);
                dim3 blocks((dst.shape[2] + 7) / 8, (dst.shape[1] + 7) / 8, (dst.shape[0] + 7) / 8);
                apply_binary_op_rank3<<<blocks, threads, 0, stream>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 4:
            {
                dim3 threads(8, 8, dst.shape[1] < 8 ? dst.shape[1] : 8);
                dim3 blocks(
                    (dst.shape[3] + threads.x - 1) / threads.x,
                    (dst.shape[2] + threads.y - 1) / threads.y,
                    dst.shape[0]
                );
                apply_binary_op_rank4<<<blocks, threads, 0, stream>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        default:
            {
                size_t total = dst.size();
                dim3 threads(256);
                dim3 blocks((total + 255) / 256);
                apply_binary_op_nd<<<blocks, threads, 0, stream>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        }
        CUDA_CHECK;
        if (stream == nullptr) cudaDeviceSynchronize();
    }

#define INSTANTIATE_BINARY_FUSED(T) \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryAdd<T>,                          cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinarySub<T>,                          cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryMul<T>,                          cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryDiv<T>,                          cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryCompose<BinaryAdd<T>, Mul<T>>,   cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryCompose<BinarySub<T>, Mul<T>>,   cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryCompose<BinaryMul<T>, Add<T>>,   cudaStream_t); \
    template void launch_apply_binary_op<T>(const TensorView<const T>, const TensorView<const T>, TensorView<T>, BinaryCompose<BinaryDiv<T>, Add<T>>,   cudaStream_t);

    INSTANTIATE_BINARY_FUSED(float)
    INSTANTIATE_BINARY_FUSED(int)
    INSTANTIATE_BINARY_FUSED(char)
    INSTANTIATE_BINARY_FUSED(float16_t)

    // Explicit instantiations — Add
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Add<float>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Add<int>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Add<char>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Add<float16_t>, cudaStream_t);

    // Explicit instantiations — Mul
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Mul<float>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Mul<int>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Mul<char>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Mul<float16_t>, cudaStream_t);

    // Explicit instantiations — Div
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Div<float>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Div<int>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Div<char>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Div<float16_t>, cudaStream_t);

    // Explicit instantiations — scale_shift: Compose<Mul, Add>
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Compose<Mul<float>,     Add<float>>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Compose<Mul<int>,       Add<int>>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Compose<Mul<char>,      Add<char>>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Compose<Mul<float16_t>, Add<float16_t>>, cudaStream_t);

    // Explicit instantiations — shift_scale: Compose<Add, Mul>
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Compose<Add<float>,     Mul<float>>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Compose<Add<int>,       Mul<int>>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Compose<Add<char>,      Mul<char>>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Compose<Add<float16_t>, Mul<float16_t>>, cudaStream_t);

    // Explicit instantiations — ReLU
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     ReLU<float>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       ReLU<int>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      ReLU<char>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, ReLU<float16_t>, cudaStream_t);

    // Explicit instantiations — Sigmoid
    template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Sigmoid<float>,     cudaStream_t);
    template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Sigmoid<int>,       cudaStream_t);
    template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Sigmoid<char>,      cudaStream_t);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Sigmoid<float16_t>, cudaStream_t);
}
