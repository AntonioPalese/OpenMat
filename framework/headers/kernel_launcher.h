#pragma once

#include "mat_utils.h"
#include "ops/kernels/fill_gpu.cuh"
#include "ops/cpu/fill_cpu.h"
// #include "kernels/add.cuh"

namespace om {
    template<DEVICE_TYPE Device, typename T>
    struct fill_dispatch;

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CPU, T> {
        static void exec(MatView<T> mat, T value) {
            fill_cpu(mat, value);
        }
    };

    template<typename T>
    struct fill_dispatch<DEVICE_TYPE::CUDA, T> {
        static void exec(MatView<T> mat, T value) {
            launch_fill(mat, value);
        }
    };
}

#include "kernel_launcher.inl"
