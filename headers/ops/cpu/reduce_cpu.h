#pragma once
#include "tensor_view.cuh"
#include <stdexcept>
#include <limits>

namespace om
{
    template<typename T>
    T reduce_sum_cpu(const TensorView<const T> src) {
        T acc = static_cast<T>(0);
        size_t n = src.size();
        for (size_t i = 0; i < n; ++i) acc = acc + src[i];
        return acc;
    }

    template<typename T>
    T reduce_min_cpu(const TensorView<const T> src) {
        if (src.size() == 0) throw std::invalid_argument("reduce_min: empty tensor");
        T acc = src[0];
        size_t n = src.size();
        for (size_t i = 1; i < n; ++i) if (src[i] < acc) acc = src[i];
        return acc;
    }

    template<typename T>
    T reduce_max_cpu(const TensorView<const T> src) {
        if (src.size() == 0) throw std::invalid_argument("reduce_max: empty tensor");
        T acc = src[0];
        size_t n = src.size();
        for (size_t i = 1; i < n; ++i) if (src[i] > acc) acc = src[i];
        return acc;
    }
}
