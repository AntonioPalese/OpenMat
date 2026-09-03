#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <stdexcept>
#include "ops/div_policy.h"

// Same size-gated parallel-for as DEFINE_BINARY_OPS_CPU (headers/ops/cpu/binary_op_macros.h)
// — see that file for why the threshold and the _Pragma indirection are needed.
#define DEFINE_UNARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(const TensorView<const T> lhs, T value, TensorView<T> dst) {\
        static_assert(is_extended_arithmetic<T>{}, "unary op requires an arithmetic type");\
        size_t _total = lhs.size();\
        _Pragma("omp parallel for schedule(static) if(_total > 65536)")\
        for(size_t idx = 0; idx < _total; ++idx)\
            dst[idx] = OP_EXPR;\
    }

namespace om 
{
    DEFINE_UNARY_OPS_CPU(add_k, lhs[idx] + value)
    DEFINE_UNARY_OPS_CPU(sub_k, lhs[idx] - value)
    DEFINE_UNARY_OPS_CPU(mul_k, lhs[idx] * value)
    DEFINE_UNARY_OPS_CPU(div_k, div_elem(lhs[idx], value))
}
