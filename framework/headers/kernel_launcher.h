#pragma once

#include "mat_utils.h"
#include "ops/kernels/fill_gpu.cuh"
#include "ops/kernels/binary_op_macros.cuh"
#include "ops/cpu/fill_cpu.h"
#include "ops/cpu/binary_op_macros.h"

// #define DEFINE_DEVICE_DISPATCH_BINARY_INL(OP_NAME)                                                                                                                                                                                                               \
//         template<typename T>                                                                  \
//         inline void _##OP_NAME(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst, DEVICE_TYPE dev) { \
//             switch (dev) {                                                                    \
//                 case DEVICE_TYPE::CPU:  OP_NAME##_dispatch<DEVICE_TYPE::CPU, T>::exec(lhs, rhs, dst); break; \
//                 case DEVICE_TYPE::CUDA: OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T>::exec(lhs, rhs, dst); break; \
//             }                                                                                 \
//         }                                                                                     

// #define DEFINE_DEVICE_DISPATCH_BINARY_H(OP_NAME, CPU_FUNC, CUDA_FUNC)                         \
//         template<DEVICE_TYPE Device, typename T>                                              \
//         struct OP_NAME##_dispatch;                                                            \
//                                                                                               \
//         template<typename T>                                                                  \
//         struct OP_NAME##_dispatch<DEVICE_TYPE::CPU, T> {                                      \
//             static void exec(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {                \
//                 CPU_FUNC(lhs, rhs, dst);                                                      \
//             }                                                                                 \
//         };                                                                                    \
//                                                                                               \
//         template<typename T>                                                                  \
//         struct OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T> {                                     \
//             static void exec(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {                \
//                 CUDA_FUNC(lhs, rhs, dst);                                                     \
//             }                                                                                 \
//         };                                                                                    

namespace om 
{
    template<DEVICE_TYPE Device, typename T>
    struct fill_dispatch;

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CPU, T> {
        static void exec(TensorView<T>& tensor, T value) {
            fill_cpu(tensor, value);
        }
    };

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CUDA, T> {
        static void exec(TensorView<T>& tensor, T value) {
            launch_fill(tensor, value);
        }
    };

    // DEFINE_DEVICE_DISPATCH_BINARY_H(add, add_cpu, launch_add)
    // DEFINE_DEVICE_DISPATCH_BINARY_H(sub, sub_cpu, launch_sub)
    // DEFINE_DEVICE_DISPATCH_BINARY_H(mul, mul_cpu, launch_mul)
    // DEFINE_DEVICE_DISPATCH_BINARY_H(div, div_cpu, launch_div)
}

#include "kernel_launcher.inl"
