#pragma once
#include "tensor_view.cuh"

namespace om 
{
    template<typename T>
    void fill_cpu(TensorView<T> tensor, T value) 
    {
        size_t _total = tensor.size();
        for(size_t idx = 0; idx < _total; ++idx)
            tensor[idx] = value;
    };
}
