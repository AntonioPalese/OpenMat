#pragma once
#include "mat_view.cuh"
#include <type_traits>
#include <stdexcept>
#include <limits>

#define DEFINE_BINARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(MatView<const T> lhs, MatView<const T> rhs, MatView<T> dst) {\
        static_assert(std::is_arithmetic_v<T>, "add_cpu requires an arithmetic type");\
\
        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {\
            throw std::runtime_error("Matrix dimensions must match for arithmetic OP");\
        }\
\
        for (int r = 0; r < lhs.rows; ++r) {\
            for (int c = 0; c < lhs.cols; ++c) {\
                dst(r, c) = OP_EXPR;\
            }\
        }\
    }

namespace om 
{
    DEFINE_BINARY_OPS_CPU(add, lhs(r, c) + rhs(r, c))
    DEFINE_BINARY_OPS_CPU(sub, lhs(r, c) - rhs(r, c))
    DEFINE_BINARY_OPS_CPU(mul, lhs(r, c) * rhs(r, c))
    DEFINE_BINARY_OPS_CPU(div, ( static_cast<double>(rhs(r, c)) != 0.0 ? lhs(r, c) / rhs(r, c) : std::numeric_limits<T>::infinity())) 
}
