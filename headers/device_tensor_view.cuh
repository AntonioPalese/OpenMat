#pragma once
#include <vector>
#include <cstring>
#include <cassert>

#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

namespace om {

template<typename T>
struct DeviceTensorView {

    DeviceTensorView(T* _data, const size_t* h_shape, const size_t* h_stride, size_t _rank)
        : data(_data), rank(_rank)
    {
        // Allocate and copy shape
        CUDA_CALL(cudaMalloc(&shape, sizeof(size_t) * rank));
        CUDA_CALL(cudaMemcpy(shape, h_shape, sizeof(size_t) * rank, cudaMemcpyHostToDevice));

        // Allocate and copy stride
        CUDA_CALL(cudaMalloc(&stride, sizeof(size_t) * rank));
        CUDA_CALL(cudaMemcpy(stride, h_stride, sizeof(size_t) * rank, cudaMemcpyHostToDevice));
    }

    // No copy constructor
    DeviceTensorView(const DeviceTensorView&) = delete;
    // No copy assignement
    DeviceTensorView& operator=(const DeviceTensorView&) = delete;

    // Move constructor
    DeviceTensorView(DeviceTensorView&& other) noexcept {
        *this = std::move(other);
    }
    // Move assignement
    DeviceTensorView& operator=(DeviceTensorView&& other) noexcept {
        if (this != &other) {
            free_device_metadata();
            data   = other.data;
            shape  = other.shape;
            stride = other.stride;
            rank   = other.rank;
            other.shape = nullptr;
            other.stride = nullptr;
        }
        return *this;
    }

    ~DeviceTensorView() {
        free_device_metadata();
    }

    template <typename... Indices>
    __device__
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

        constexpr size_t num_indices = sizeof...(Indices);
        assert(num_indices == rank && "Incorrect number of indices for tensor access.");

        size_t idx_array[] = { static_cast<size_t>(indices)... };
        return data[compute_flat_index(idx_array)];
    }        
    template <typename... Indices>
    __device__
    T operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

        constexpr size_t num_indices = sizeof...(Indices);
        assert(num_indices == rank && "Incorrect number of indices for tensor access.");

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
    size_t size() const
    {
        size_t acc = 1;
        for(int i = 0; i < rank; ++i)
            acc *= shape[i];
        return acc;
    }

    __device__
    size_t compute_flat_index(const size_t* indices) const {
        size_t flat = 0;
        for (size_t i = 0; i < rank; ++i) {
            flat += indices[i] * stride[i];
        }
        return flat;
    }

    T* __restrict__ data = nullptr;
    size_t* shape = nullptr;
    size_t* stride = nullptr;
    size_t rank = 0;

    void free_device_metadata() {
        if (shape) cudaFree(shape);
        if (stride) cudaFree(stride);
    }
};

} // namespace om
