#pragma once

#include "mat_utils.h"
#include "ops/kernels/fill_gpu.cuh"
#include "ops/cpu/fill_cpu.h"
// #include "kernels/add.cuh"

#define CUDA_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) \
            throw std::runtime_error(cudaGetErrorString(err)); \
    } while (0)


namespace om {

    template<DEVICE_TYPE device, typename T>
    void _fill(MatView<T> mat, T value);
    
    template<typename T>
    void _fill(MatView<T> mat, T value, Device device); // runtime switch

    // template <typename T>
    // void add(MatView<T> a, MatView<T> b, MatView<T> out) {
    //     launch_add(a, b, out);
    // }
}

#include "kernel_launcher.inl"
