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


    // Apply a unary operation \p op element-wise from \p src to \p dst.
    //
    // Kernels are defined in the corresponding .cu file; only the host
    // dispatch function is exposed here.
    template<typename T, typename Op>
    void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op);
}
