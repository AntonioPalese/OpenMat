#pragma once

#include "mat_utils.h"
#include "ops/kernels/fill_gpu.cuh"
#include "ops/kernels/binary_op_macros.cuh"
#include "ops/kernels/unary_op_macros.cuh"
#include "ops/cpu/fill_cpu.h"
#include "ops/cpu/binary_op_macros.h"
#include "ops/cpu/unary_op_macros.h"


#define DEFINE_DEVICE_DISPATCH_BINARY_INL(OP_NAME)\
        template<typename T>\
        inline void _##OP_NAME(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst, DEVICE_TYPE dev) {\
            switch (dev) {\
                case DEVICE_TYPE::CPU:  OP_NAME##_dispatch<DEVICE_TYPE::CPU, T>::exec(lhs, rhs, dst); break;\
                case DEVICE_TYPE::CUDA: OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T>::exec(lhs, rhs, dst); break;\
            }\
        }

#define DEFINE_DEVICE_DISPATCH_BINARY_H(OP_NAME, CPU_FUNC, CUDA_FUNC)\
        template<DEVICE_TYPE Device, typename T>\
        struct OP_NAME##_dispatch;\
\
        template<typename T>\
        struct OP_NAME##_dispatch<DEVICE_TYPE::CPU, T> {\
            static void exec(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {\
                CPU_FUNC(lhs, rhs, dst);\
            }\
        };\
\
        template<typename T>\
        struct OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T> {\
            static void exec(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {\
                CUDA_FUNC(lhs, rhs, dst);\
            }\
        };               
        
#define DEFINE_DEVICE_DISPATCH_UNARY_INL(OP_NAME)\
        template<typename T>\
        inline void _##OP_NAME(const TensorView<const T> lhs, T value, TensorView<T> dst, DEVICE_TYPE dev) {\
            switch (dev) {\
                case DEVICE_TYPE::CPU:  OP_NAME##_dispatch<DEVICE_TYPE::CPU, T>::exec(lhs, value, dst); break;\
                case DEVICE_TYPE::CUDA: OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T>::exec(lhs, value, dst); break;\
            }\
        }

#define DEFINE_DEVICE_DISPATCH_UNARY_H(OP_NAME, CPU_FUNC, CUDA_FUNC)\
        template<DEVICE_TYPE Device, typename T>\
        struct OP_NAME##_dispatch;\
\
        template<typename T>\
        struct OP_NAME##_dispatch<DEVICE_TYPE::CPU, T> {\
            static void exec(const TensorView<const T> lhs, T value, TensorView<T> dst) {\
                CPU_FUNC(lhs, value, dst);\
            }\
        };\
\
        template<typename T>\
        struct OP_NAME##_dispatch<DEVICE_TYPE::CUDA, T> {\
            static void exec(const TensorView<const T> lhs, T value, TensorView<T> dst) {\
                CUDA_FUNC(lhs, value, dst);\
            }\
        };                                                                                    

namespace om 
{
    template<DEVICE_TYPE Device, typename T>
    struct fill_dispatch;

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CPU, T> {
        static void exec(TensorView<T> tensor, T value) {
            fill_cpu(tensor, value);
        }
    };

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CUDA, T> {
        static void exec(TensorView<T> tensor, T value) {
            launch_fill(tensor, value);
        }
    };

    DEFINE_DEVICE_DISPATCH_BINARY_H(add, add_cpu, launch_add)
    DEFINE_DEVICE_DISPATCH_BINARY_H(sub, sub_cpu, launch_sub)
    DEFINE_DEVICE_DISPATCH_BINARY_H(mul, mul_cpu, launch_mul)
    DEFINE_DEVICE_DISPATCH_BINARY_H(div, div_cpu, launch_div)

    DEFINE_DEVICE_DISPATCH_UNARY_H(add_k, add_k_cpu, launch_add_k)
    DEFINE_DEVICE_DISPATCH_UNARY_H(sub_k, sub_k_cpu, launch_sub_k)
    DEFINE_DEVICE_DISPATCH_UNARY_H(mul_k, mul_k_cpu, launch_mul_k)
    DEFINE_DEVICE_DISPATCH_UNARY_H(div_k, div_k_cpu, launch_div_k)
}

#include "kernel_launcher.inl"
