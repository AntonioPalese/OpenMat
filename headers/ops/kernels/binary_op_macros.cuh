#pragma once
#include "mat_view.cuh"
#include "cuda_defines.cuh"
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH(OP_NAME, OP_EXPR)                                      \
    template<typename T>                                                                          \
    __global__ void OP_NAME##_kernel(MatView<const T> lhs, MatView<const T> rhs, MatView<T> dst) {            \
        int r = blockIdx.y * blockDim.y + threadIdx.y;                                             \
        int c = blockIdx.x * blockDim.x + threadIdx.x;                                             \
        if (r < dst.rows && c < dst.cols)                                                         \
            dst(r, c) = OP_EXPR;                                                                  \
    }                                                                                              \
                                                                                                   \
    template<typename T>                                                                          \
    void launch_##OP_NAME(MatView<const T> lhs, MatView<const T> rhs, MatView<T> dst) {                       \
        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols ||                                       \
            lhs.rows != dst.rows || lhs.cols != dst.cols) {                                       \
            throw std::runtime_error("Matrix size mismatch in " #OP_NAME);                        \
        }                                                                                          \
        dim3 threads(16, 16);                                                                      \
        dim3 blocks((dst.cols + 15) / 16, (dst.rows + 15) / 16);                                   \
        OP_NAME##_kernel<<<blocks, threads>>>(lhs, rhs, dst);                                     \
        CUDA_CHECK;                                                                                \
        cudaDeviceSynchronize();                                                                   \
    }

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(OP_NAME)                                         \
    template void launch_##OP_NAME<float>(MatView<const float> lhs, MatView<const float> rhs, MatView<float> dst);  \
    template void launch_##OP_NAME<int>(MatView<const int> lhs, MatView<const int> rhs, MatView<int> dst);          \
    template void launch_##OP_NAME<char>(MatView<const char> lhs, MatView<const char> rhs, MatView<char> dst);

#define DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(OP_NAME)                                             \
    template<typename T>                                                                          \
    __global__ void OP_NAME##_kernel(MatView<const T> lhs, MatView<const T> rhs, MatView<T> dst);             \
                                                                                                  \
    template<typename T>                                                                          \
    void launch_##OP_NAME(MatView<const T> lhs, MatView<const T> rhs, MatView<T> dst);                        

namespace om 
{
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(add)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(sub)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(mul)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_H(div)  // optional: guard for div-by-0
}