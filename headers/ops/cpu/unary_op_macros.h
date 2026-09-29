#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <stdexcept>
#include "ops/div_policy.h"
#include "ops/cpu/broadcast_cpu.h"

// The loop — contiguous fast path, strided row walk and the OpenMP threshold —
// lives in unary_elementwise_cpu (broadcast_cpu.h). OP_EXPR is written in
// terms of the element x and the scalar `value`.
#define DEFINE_UNARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(const TensorView<const T> lhs, T value, TensorView<T> dst) {\
        static_assert(is_extended_arithmetic<T>{}, "unary op requires an arithmetic type");\
        ::om::detail::unary_elementwise_cpu(lhs, dst,\
            [value](const T& x) -> T { return OP_EXPR; });\
    }

namespace om
{
    DEFINE_UNARY_OPS_CPU(add_k, x + value)
    DEFINE_UNARY_OPS_CPU(sub_k, x - value)
    DEFINE_UNARY_OPS_CPU(mul_k, x * value)
    DEFINE_UNARY_OPS_CPU(div_k, div_elem(x, value))
}
