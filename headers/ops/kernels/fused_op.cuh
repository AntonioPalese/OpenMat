#pragma once
#include "device_tensor_view.cuh"
#include "tensor_view.cuh"

namespace om 
{

    template <typename T>
    struct Add {
        T a;
        __device__ T operator()(T x) const { return x + a; }
    };

    template <typename T>
    struct Mul {
        T b;
        __device__ T operator()(T x) const { return x * b; }
    };

    template <typename F, typename G>
    struct Compose {
        F f;
        G g;
        __device__ auto operator()(auto x) const { return g(f(x)); }
    };


    template<typename T>
    __global__ void apply_op_rank1(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void apply_op_rank2(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void apply_op_rank3(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void apply_op_rank4(DeviceTensorView<T> tensor, T value);
    template<typename T>
    __global__ void apply_op_nd(DeviceTensorView<T> tensor, T value);

    template<typename T>
    void launch_apply_op(TensorView<T> tensor, T value);
}
