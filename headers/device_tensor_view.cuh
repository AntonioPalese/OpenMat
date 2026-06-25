#pragma once
#include <cassert>

#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

namespace om {

constexpr size_t MAX_RANK = 8;

template<typename T>
struct DeviceTensorView {

    __host__
    DeviceTensorView(T* _data, const size_t* h_shape, const size_t* h_stride, size_t _rank)
        : data(_data), rank(_rank)
    {
        assert(_rank <= MAX_RANK && "Tensor rank exceeds MAX_RANK (8)");
        for (size_t i = 0; i < _rank; ++i) {
            shape[i]  = h_shape[i];
            stride[i] = h_stride[i];
        }
    }

    DeviceTensorView()                                     = default;
    DeviceTensorView(const DeviceTensorView&)              = default;
    DeviceTensorView& operator=(const DeviceTensorView&)   = default;
    DeviceTensorView(DeviceTensorView&&)                   = default;
    DeviceTensorView& operator=(DeviceTensorView&&)        = default;
    ~DeviceTensorView()                                    = default;

    template <typename... Indices>
    __device__
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");
        size_t idx_array[] = { static_cast<size_t>(indices)... };
        return data[compute_flat_index(idx_array)];
    }

    template <typename... Indices>
    __device__
    T operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");
        size_t idx_array[] = { static_cast<size_t>(indices)... };
        return device_load(&data[compute_flat_index(idx_array)]);
    }

    __device__
    T& operator[](size_t flat_index) {
        return data[flat_index];
    }

    __device__
    const T& operator[](size_t flat_index) const {
        return data[flat_index];
    }

    __device__
    size_t size() const {
        size_t acc = 1;
        for (size_t i = 0; i < rank; ++i)
            acc *= shape[i];
        return acc;
    }

    __device__
    size_t compute_flat_index(const size_t* indices) const {
        size_t flat = 0;
        for (size_t i = 0; i < rank; ++i)
            flat += indices[i] * stride[i];
        return flat;
    }

    T* __restrict__ data   = nullptr;
    size_t shape[MAX_RANK] = {};
    size_t stride[MAX_RANK] = {};
    size_t rank             = 0;
};

} // namespace om
