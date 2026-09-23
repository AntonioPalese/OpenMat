#pragma once
#include <cstddef>
#include <stdexcept>
#include "tensor_view.cuh"

namespace om::detail
{
    // The one CPU loop behind add/sub/mul/div and apply_binary.
    //
    // lhs and rhs arrive already expanded to dst's shape (see broadcast.h): a
    // broadcast operand carries stride 0 on its broadcast axes. When all three
    // are contiguous — every same-shape call — this is the flat loop the ops
    // always ran. Otherwise it walks the output row by row: all axes but the
    // last pick a row, whose base offsets are decoded once, and the last axis
    // is a constant-stride inner loop (stride 0 or 1 for the usual bias case),
    // which the compiler can vectorize.
    //
    // Fork/join plus the barrier at the end of a parallel region cost on the
    // order of a microsecond even with OpenMP's persistent thread pool —
    // comparable to or larger than the loop itself below this many elements,
    // so the loop is only handed to the team above a size threshold (the "if"
    // clause) rather than paying that tax on every call. _Pragma rather than
    // #pragma keeps the spelling usable from inside a macro as well.
    template <typename T, typename Op>
    void binary_elementwise_cpu(const TensorView<const T> lhs, const TensorView<const T> rhs,
                                TensorView<T> dst, Op op)
    {
        if (!lhs.same_shape(dst) || !rhs.same_shape(dst))
            throw std::runtime_error("Tensor dimensions must match for arithmetic operations");

        const size_t total = dst.size();
        if (total == 0) return;

        if (lhs.is_contiguous() && rhs.is_contiguous() && dst.is_contiguous()) {
            _Pragma("omp parallel for schedule(static) if(total > 65536)")
            for (size_t i = 0; i < total; ++i)
                dst[i] = op(lhs[i], rhs[i]);
            return;
        }

        const size_t last  = dst.rank - 1;
        const size_t inner = dst.shape[last];
        const size_t rows  = total / inner;
        const size_t ls = lhs.stride[last], rs = rhs.stride[last], ds = dst.stride[last];

        _Pragma("omp parallel for schedule(static) if(total > 65536)")
        for (size_t row = 0; row < rows; ++row) {
            size_t lo = 0, ro = 0, doff = 0, tmp = row;
            for (size_t d = last; d-- > 0; ) {
                const size_t coord = tmp % dst.shape[d];
                tmp /= dst.shape[d];
                lo   += coord * lhs.stride[d];
                ro   += coord * rhs.stride[d];
                doff += coord * dst.stride[d];
            }
            for (size_t j = 0; j < inner; ++j)
                dst.data[doff + j * ds] = op(lhs.data[lo + j * ls], rhs.data[ro + j * rs]);
        }
    }
}
