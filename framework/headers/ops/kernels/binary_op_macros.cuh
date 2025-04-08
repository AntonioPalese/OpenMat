#pragma once
#include "mat_view.cuh"
#include "cuda_defines.cuh"
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_K1(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank1(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {\
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;\
\
        if (x < lhs.shape[0] && x < rhs.shape[0])\
            dst(x) = OP_EXPR;\
    }\
\
    template<typename T>\
    void launch_##OP_NAME(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {\
        if ( !lhs.match(dst) || !rhs.match(dst) || !lhs.match(rhs) )\
        {\
            throw std::runtime_error("Matrix size mismatch in " #OP_NAME);\
        }\
        dim3 threads(16);\
        dim3 blocks((dst.shape[0] + 15) / 16);\
        OP_NAME##_kernel<<<blocks, threads>>>(lhs, rhs, dst);\
        CUDA_CHECK;\
        cudaDeviceSynchronize();\
    }

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_K2(OP_NAME, OP_EXPR)\
    template<typename T>\
    __global__ void OP_NAME##_kernel_rank2(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {\
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;\
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;\
\
        if (y < lhs.shape[1] && y < rhs.shape[1] && x < lhs.shape[0] && x < rhs.shape[0])\
            dst(y, x) = OP_EXPR;\
    }\
\
    template<typename T>\
    void launch_##OP_NAME(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {\
        if ( !lhs.match(dst) || !rhs.match(dst) || !lhs.match(rhs) )\
        {\
            throw std::runtime_error("Matrix size mismatch in " #OP_NAME);\
        }\
        dim3 threads(16, 16);\
        dim3 blocks((dst.shape[1] + 15) / 16, (dst.shape[0] + 15) / 16);\
        OP_NAME##_kernel<<<blocks, threads>>>(lhs, rhs, dst);\
        CUDA_CHECK;\
        cudaDeviceSynchronize();\
    }

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(OP_NAME)                                         \
    template void launch_##OP_NAME<float>(TensorView<const float> lhs, TensorView<const float> rhs, TensorView<float> dst);  \
    template void launch_##OP_NAME<int>(TensorView<const int> lhs, TensorView<const int> rhs, TensorView<int> dst);          \
    template void launch_##OP_NAME<char>(TensorView<const char> lhs, TensorView<const char> rhs, TensorView<char> dst);

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(OP_NAME)                                             \
    template<typename T>                                                                          \
    __global__ void OP_NAME##_kernel(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst);             \
                                                                                                  \
    template<typename T>                                                                          \
    void launch_##OP_NAME(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst);                        

namespace om 
{
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(add)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(sub)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(mul)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(div)  // optional: guard for div-by-0
}