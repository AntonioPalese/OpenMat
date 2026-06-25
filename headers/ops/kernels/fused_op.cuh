#pragma once
#include <cmath>
#include "device_tensor_view.cuh"
#include "tensor_view.cuh"

namespace om 
{

    template <typename T>
    struct Add {
        T a;
        __host__ __device__ T operator()(T x) const { return x + a; }
    };

    template <typename T>
    struct Mul {
        T b;
        __host__ __device__ T operator()(T x) const { return x * b; }
    };

    template <typename T>
    struct Div {
        T b;
        __host__ __device__ T operator()(T x) const { return x / b; }
    };

    template <typename T>
    struct Pow {
        T b;
        __host__ __device__ T operator()(T x) const { return pow(x, b); }
    };

    template <typename F, typename G>
    struct Compose {
        F f;
        G g;

        template <typename T>
        __host__ __device__ auto operator()(T x) const -> decltype(g(f(x))) { return g(f(x)); }
    };

    template<typename T, typename Op>
    void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op);

    // ---------------------------------------------------------------------------
    // Binary fused functors: Op(lhs[i], rhs[i])
    // ---------------------------------------------------------------------------

    template <typename T>
    struct BinaryAdd {
        __host__ __device__ T operator()(T x, T y) const { return x + y; }
    };

    template <typename T>
    struct BinarySub {
        __host__ __device__ T operator()(T x, T y) const { return x - y; }
    };

    template <typename T>
    struct BinaryMul {
        __host__ __device__ T operator()(T x, T y) const { return x * y; }
    };

    template <typename T>
    struct BinaryDiv {
        __host__ __device__ T operator()(T x, T y) const { return x / y; }
    };

    // Compose a binary op with a unary post-op: dst[i] = post(bin(lhs[i], rhs[i]))
    template <typename BinOp, typename UnaryOp>
    struct BinaryCompose {
        BinOp bin;
        UnaryOp post;

        template <typename T>
        __host__ __device__ auto operator()(T x, T y) const -> decltype(post(bin(x, y))) {
            return post(bin(x, y));
        }
    };

    template<typename T, typename Op>
    void launch_apply_binary_op(const TensorView<const T> lhs, const TensorView<const T> rhs,
                                TensorView<T> dst, Op op);

    template <typename T>
    struct ReLU {
        __host__ __device__ T operator()(T x) const {
            return static_cast<float>(x) > 0.0f ? x : static_cast<T>(0);
        }
    };

    template <typename T>
    struct Sigmoid {
        __host__ __device__ T operator()(T x) const {
            return static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(expf(-static_cast<float>(x))));
        }
    };
}
