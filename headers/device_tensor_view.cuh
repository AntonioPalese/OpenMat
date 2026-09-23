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

    // The offset is a fold over the pack, not a loop over `rank` reading a
    // local index array. The pack's length is a compile-time constant, so
    // this unrolls into multiply-adds on registers. The array form could not:
    // indexed by a loop whose bound is the *runtime* rank, the array was
    // placed in local memory (a 16-byte stack frame), and every element of
    // every rank-specialized kernel paid a local store and reload for it.
    // Measured on (4096,4096)+(4096,) fp32: 880 µs → 537 µs, i.e. 153 →
    // 250 GB/s, which is the copy ceiling of the machine.
    template <typename... Indices>
    __device__
    size_t offset_of(Indices... indices) const {
        static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");
        size_t flat = 0;
        size_t d = 0;
        ((flat += static_cast<size_t>(indices) * stride[d++]), ...);
        return flat;
    }

    template <typename... Indices>
    __device__
    T& operator()(Indices... indices) {
        return data[offset_of(indices...)];
    }

    template <typename... Indices>
    __device__
    T operator()(Indices... indices) const {
        return device_load(&data[offset_of(indices...)]);
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
