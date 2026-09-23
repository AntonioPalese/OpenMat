#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <type_traits>
#include <stdexcept>
#include "ops/div_policy.h"

#include "ops/cpu/broadcast_cpu.h"

// The loop itself — contiguous fast path, broadcast/strided path and the
// OpenMP threshold — lives in binary_elementwise_cpu (broadcast_cpu.h), shared
// with apply_binary. OP_EXPR is written in terms of the two elements a and b.
#define DEFINE_BINARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {\
        static_assert(is_extended_arithmetic<T>::value, "binary op requires an arithmetic type");\
        ::om::detail::binary_elementwise_cpu(lhs, rhs, dst,\
            [](const T& a, const T& b) -> T { return OP_EXPR; });\
    }

namespace om 
{
    DEFINE_BINARY_OPS_CPU(add, a + b)
    DEFINE_BINARY_OPS_CPU(sub, a - b)
    DEFINE_BINARY_OPS_CPU(mul, a * b)
    DEFINE_BINARY_OPS_CPU(div, div_elem(a, b))
}
