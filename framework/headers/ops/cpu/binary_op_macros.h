// #pragma once
// #include "tensor_view.cuh"
// #include <type_traits>
// #include <stdexcept>
// #include <limits>

// #define DEFINE_BINARY_OPS_CPU(OP_NAME, OP_EXPR)\
//     template<typename T>\
//     void OP_NAME##_cpu(TensorView<const T> lhs, TensorView<const T> rhs, TensorView<T> dst) {\
//         static_assert(std::is_arithmetic_v<T>, "add_cpu requires an arithmetic type");\
// \
//         if (!lhs.match(rhs)) {\
//             throw std::runtime_error("Tensor dimensions must match for arithmetic operations");\
//         }\
// \
//         size_t _total = lhs.size();\
//         for(size_t idx = 0; idx < _total; ++idx)\
//             dst[idx] = OP_EXPR;\
//     }

// namespace om 
// {
//     DEFINE_BINARY_OPS_CPU(add, lhs[idx] + rhs[idx])
//     DEFINE_BINARY_OPS_CPU(sub, lhs[idx] - rhs[idx])
//     DEFINE_BINARY_OPS_CPU(mul, lhs[idx] * rhs[idx])
//     DEFINE_BINARY_OPS_CPU(div, ( static_cast<double>(rhs[idx]) != 0.0 ? lhs[idx] / rhs[idx] : std::numeric_limits<T>::infinity()))
// }
