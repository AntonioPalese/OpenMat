#pragma once
#include "tensor_view.cuh"
#include <stdexcept>
#include <vector>

namespace om
{

template<typename T>
inline void transpose_cpu(const TensorView<const T> src, TensorView<T> dst)
{
    if (src.rank != 2 || dst.rank != 2)
        throw std::runtime_error("transpose_cpu: tensors must be rank-2");
    if (src.shape[0] != dst.shape[1] || src.shape[1] != dst.shape[0])
        throw std::runtime_error("transpose_cpu: dst shape must be transposed src shape");

    size_t M = src.shape[0];
    size_t N = src.shape[1];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            dst(j, i) = src(i, j);
}

template<typename T>
inline void permute_cpu(const TensorView<const T> src, TensorView<T> dst,
                        const size_t* axes, size_t rank)
{
    size_t total = dst.size();

    // Iterate over all dst flat indices
    // We'll walk through each output element by reconstructing its multi-index.
    std::vector<size_t> dst_idx(rank);
    std::vector<size_t> src_idx(rank);

    for (size_t flat = 0; flat < total; ++flat) {
        // Decompose flat dst index
        size_t tmp = flat;
        for (int d = (int)rank - 1; d >= 0; --d) {
            dst_idx[d] = tmp % dst.shape[d];
            tmp        /= dst.shape[d];
        }

        // Map to src: src_idx[axes[d]] = dst_idx[d]
        for (size_t d = 0; d < rank; ++d)
            src_idx[axes[d]] = dst_idx[d];

        // Compute flat src offset using src strides
        size_t src_flat = 0;
        for (size_t d = 0; d < rank; ++d)
            src_flat += src_idx[d] * src.stride[d];

        dst[flat] = src.data[src_flat];
    }
}

} // namespace om
