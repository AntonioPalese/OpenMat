#pragma once
#include "device_tensor_view.cuh"
#include "tensor_view.cuh"

namespace om 
{
    template<typename T>
    __global__ void fill_kernel_rank1(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void fill_kernel_rank2(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void fill_kernel_rank3(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void fill_kernel_rank4(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void fill_kernel_nd(DeviceTensorView<T> tensor, T value);

    template<typename T>
    void launch_fill(TensorView<T> tensor, T value);
}
