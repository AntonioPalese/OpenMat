#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <type_traits>
#include <stdexcept>
#include "ops/div_policy.h"

// Fork/join plus the barrier at the end of a parallel region cost on the
// order of a microsecond even with OpenMP's persistent thread pool (no OS
// thread creation after the first region) — comparable to or larger than
// the scalar loop itself below this many elements, so the loop is only
// handed to the team above a size threshold (the "if" clause) rather than
// paying that tax on every call, including the ones on the hot path for
// small tensors. _Pragma is required rather than a plain #pragma line
// because this sits inside a macro replacement list; #pragma cannot appear
// mid-macro, _Pragma can, and it is expanded at the same point.
#define DEFINE_BINARY_OPS_CPU(OP_NAME, OP_EXPR)\
    template<typename T>\
    void OP_NAME##_cpu(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {\
        static_assert(is_extended_arithmetic<T>::value, "binary op requires an arithmetic type");\
\
        if (!lhs.match(rhs)) {\
            throw std::runtime_error("Tensor dimensions must match for arithmetic operations");\
        }\
\
        size_t _total = lhs.size();\
        _Pragma("omp parallel for schedule(static) if(_total > 65536)")\
        for(size_t idx = 0; idx < _total; ++idx)\
            dst[idx] = OP_EXPR;\
    }

namespace om 
{
    DEFINE_BINARY_OPS_CPU(add, lhs[idx] + rhs[idx])
    DEFINE_BINARY_OPS_CPU(sub, lhs[idx] - rhs[idx])
    DEFINE_BINARY_OPS_CPU(mul, lhs[idx] * rhs[idx])
    DEFINE_BINARY_OPS_CPU(div, div_elem(lhs[idx], rhs[idx]))
}
