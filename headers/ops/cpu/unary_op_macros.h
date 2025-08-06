#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <stdexcept>
#include <limits>

#define DEFINE_UNARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(const TensorView<const T> lhs, T value, TensorView<T> dst) {\
        static_assert(is_extended_arithmetic<T>{}, "unary op requires an arithmetic type");\
        size_t _total = lhs.size();\
        for(size_t idx = 0; idx < _total; ++idx)\
            dst[idx] = OP_EXPR;\
    }

namespace om 
{
    DEFINE_UNARY_OPS_CPU(add_k, lhs[idx] + value)
    DEFINE_UNARY_OPS_CPU(sub_k, lhs[idx] - value)
    DEFINE_UNARY_OPS_CPU(mul_k, lhs[idx] * value)
    DEFINE_UNARY_OPS_CPU(div_k, ( static_cast<double>(value) != 0.0 ? lhs[idx] / value : std::numeric_limits<T>::infinity()))
}
