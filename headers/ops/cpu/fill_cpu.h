#pragma once
#include "tensor_view.cuh"

namespace om
{
    template<typename T>
    void fill_cpu(TensorView<T> tensor, T value)
    {
        size_t _total = tensor.size();
        if (tensor.is_contiguous()) {
            for(size_t idx = 0; idx < _total; ++idx)
                tensor[idx] = value;
            return;
        }
        // A strided view (a slice, a select): walk it by coordinates.
        for (size_t flat = 0; flat < _total; ++flat) {
            size_t off = 0, tmp = flat;
            for (size_t d = tensor.rank; d-- > 0; ) {
                off += (tmp % tensor.shape[d]) * tensor.stride[d];
                tmp /= tensor.shape[d];
            }
            tensor.data[off] = value;
        }
    };
}
