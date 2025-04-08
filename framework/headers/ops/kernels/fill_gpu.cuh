#pragma once
#include "tensor_view.cuh"

namespace om {

    template<typename T>
    __global__ void fill_kernel_rank1(TensorView<T> mat, T value);
    template<typename T>
    __global__ void fill_kernel_rank2(TensorView<T> mat, T value);
    template<typename T>
    __global__ void fill_kernel_rank3(TensorView<T> mat, T value);
    template<typename T>
    __global__ void fill_kernel_rank4(TensorView<T> mat, T value);
    template<typename T>
    __global__ void fill_kernel_nd(TensorView<T> mat, T value);

    template<typename T>
    void launch_fill(TensorView<T> mat, T value);
}
