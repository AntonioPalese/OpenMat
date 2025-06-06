#pragma once
#include "type_traits/types.cuh"
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include "cuda_defines.cuh"
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>


#define DEFINE_BINARY_OP_LAUNCH(OP_NAME)\
    template<typename T>\
    void launch_##OP_NAME(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst)\
    {\
        if ( !lhs.match(dst) || !rhs.match(dst) || !lhs.match(rhs) )\
        {\
            throw std::runtime_error("Matrix size mismatch in " #OP_NAME);\
        }\
        switch (lhs.rank)\
        {\
        case 1:\
            {\
                dim3 threads(16);\
                dim3 blocks((lhs.shape[0] + 15) / 16);\
                OP_NAME##_kernel_rank1<<<blocks, threads>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw());\
            }\
            break;\
        case 2:\
            {\
                dim3 threads(16, 16);\
                dim3 blocks((lhs.shape[1] + 15) / 16, (lhs.shape[0] + 15) / 16);\
                OP_NAME##_kernel_rank2<<<blocks, threads>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw());\
            }\
            break;\
        case 3:\
            {\
                dim3 threads(8, 8, 8);\
                dim3 blocks((lhs.shape[2] + 7) / 8, (lhs.shape[1] + 7) / 8, (lhs.shape[0] + 7) / 8);\
                OP_NAME##_kernel_rank3<<<blocks, threads>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw());\
            }\
            break;\
        case 4:\
            {\
                dim3 threads(8, 8, lhs.shape[1] < 8 ? lhs.shape[1] : 8);\
                dim3 blocks(\
                    (lhs.shape[3] + threads.x - 1) / threads.x,\
                    (lhs.shape[2] + threads.y - 1) / threads.y,\
                    lhs.shape[0]\
                );\
                OP_NAME##_kernel_rank4<<<blocks, threads>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw());\
            }\
            break;\
        default:\
            {\
                size_t total_elements = lhs.size();\
                dim3 threads(256);\
                dim3 blocks((total_elements + threads.x - 1) / threads.x);\
                OP_NAME##_kernel_nd<<<blocks, threads>>>(lhs.as_device_tw(), rhs.as_device_tw(), dst.as_device_tw());\
            }\
            break;\
        }\
        CUDA_CHECK;\
        cudaDeviceSynchronize();\
    }

#define DEFINE_BINARY_OP_KERNEL_K1(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank1(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst) {\
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;\
\
        if (x < lhs.shape[0] && x < rhs.shape[0])\
            dst(x) = OP_EXPR;\
    }

#define DEFINE_BINARY_OP_KERNEL_K2(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank2(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst) {\
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;\
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;\
\
        if (y < lhs.shape[0] && y < rhs.shape[0] && x < lhs.shape[1] && x < rhs.shape[1])\
            dst(y, x) = OP_EXPR;\
    }

#define DEFINE_BINARY_OP_KERNEL_K3(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank3(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst) {\
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;\
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;\
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;\
\
        if (z < lhs.shape[0] && z < rhs.shape[0] && y < lhs.shape[1] && y < rhs.shape[1] && x < lhs.shape[2] && x < rhs.shape[2])\
            dst(z, y, x) = OP_EXPR;\
    }

#define DEFINE_BINARY_OP_KERNEL_K4(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank4(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst) {\
        size_t w = threadIdx.x + blockIdx.x * blockDim.x;\
        size_t h = threadIdx.y + blockIdx.y * blockDim.y;\
        size_t n = blockIdx.z;\
\
        for (size_t c = threadIdx.z; c < lhs.shape[1]; c += blockDim.z) {\
            if (n < lhs.shape[0] && n < rhs.shape[0] &&\
                c < lhs.shape[1] && c < rhs.shape[1] &&\
                h < lhs.shape[2] && h < rhs.shape[2] &&\
                w < lhs.shape[3] && w < rhs.shape[3]) {\
                    dst(n, c, h, w) = OP_EXPR;\
            }\
        }\
    }   

#define DEFINE_BINARY_OP_KERNEL_ND(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_nd(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst) {\
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\
        size_t total_elements = lhs.size();\
\
        if (idx >= total_elements) return;\
\
        size_t offset = 0;\
        size_t tmp = idx;\
        for (size_t d = 0; d < lhs.rank; ++d) {\
            size_t coord = tmp % lhs.shape[d];\
            offset += coord * lhs.stride[d];\
            tmp /= lhs.shape[d];\
        }\
\
        dst[offset] = OP_EXPR;\
    }

#define DEFINE_BINARY_OP_LAUNCH_FRW_DEC(OP_NAME)\
    template void launch_##OP_NAME<float>(const TensorView<const float> lhs, const TensorView<const float> rhs, TensorView<float> dst);\
    template void launch_##OP_NAME<int>(const TensorView<const int> lhs, const TensorView<const int> rhs, TensorView<int> dst);\
    template void launch_##OP_NAME<char>(const TensorView<const char> lhs, const TensorView<const char> rhs, TensorView<char> dst);\
    template void launch_##OP_NAME<float16_t>(const TensorView<const float16_t> lhs, const TensorView<const float16_t> rhs, TensorView<float16_t> dst);

#define DEFINE_BINARY_OP_KERNEL_H(OP_NAME)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank1(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst);\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank2(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst);\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank3(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst);\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank4(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst);\
    template<typename T>\
    __global__ void OP_NAME##_kernel_nd(const DeviceTensorView<const T> lhs, const DeviceTensorView<const T> rhs, DeviceTensorView<T> dst);

#define DEFINE_BINARY_OP_LAUNCH_H(OP_NAME)\
    template<typename T>\
    void launch_##OP_NAME(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst);                        

namespace om 
{
    DEFINE_BINARY_OP_LAUNCH_H(add);
    DEFINE_BINARY_OP_KERNEL_H(add);

    DEFINE_BINARY_OP_LAUNCH_H(sub);
    DEFINE_BINARY_OP_KERNEL_H(sub);

    DEFINE_BINARY_OP_LAUNCH_H(mul);
    DEFINE_BINARY_OP_KERNEL_H(mul);

    DEFINE_BINARY_OP_LAUNCH_H(div);
    DEFINE_BINARY_OP_KERNEL_H(div);
}