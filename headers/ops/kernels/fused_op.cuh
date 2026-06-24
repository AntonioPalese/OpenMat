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

    template <typename T>
    struct Div {
        T b;
        __device__ T operator()(T x) const { return x / b; }
    };

    template <typename T>
    struct Pow {
        T b;
        __device__ T operator()(T x) const { return pow(x, b); }
    };

    template <typename F, typename G>
    struct Compose {
        F f;
        G g;

        template <typename T>
        __device__ auto operator()(T x) const -> decltype(g(f(x))) { return g(f(x)); }
    };

    template<typename T, typename Op>
    void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op);
}
