#pragma once
#include "cuda_defines.cuh"
#include <vector>
#include <assert.h>
#include <type_traits>
#include <numeric>

namespace om {
    template<typename T>
    struct TensorView {
        T* data;
        const size_t* host_shape;
        size_t* device_shape;

        const size_t* host_stride;
        size_t* device_stride;
        
        size_t rank;

        TensorView(T* _data, const size_t* h_shape, const size_t* h_stride, size_t _rank)
        : data(_data), host_shape(h_shape), host_stride(h_stride), rank(_rank)
        {
            // Allocate and copy shape
            CUDA_CALL(cudaMalloc(&device_shape, sizeof(size_t) * rank));
            CUDA_CALL(cudaMemcpy(device_shape, host_shape, sizeof(size_t) * rank, cudaMemcpyHostToDevice));

            // Allocate and copy stride
            CUDA_CALL(cudaMalloc(&device_stride, sizeof(size_t) * rank));
            CUDA_CALL(cudaMemcpy(device_stride, host_stride, sizeof(size_t) * rank, cudaMemcpyHostToDevice));
        }

        // No copy constructor
        //TensorView(const TensorView&) = delete;
        TensorView(const TensorView& other) 
        {
            rank = other.rank;
            data = other.data;
            host_shape = other.host_shape;
            host_stride = other.host_stride;
            device_shape = other.device_shape;
            device_stride = other.device_stride;
        }
            
        TensorView& operator=(const TensorView&) = delete;

        // Move constructor
        TensorView(TensorView&& other) noexcept {
            *this = std::move(other);
        }
        TensorView& operator=(TensorView&& other) = delete;

        ~TensorView() {
            free_device_metadata();
        }

        __host__
        bool match(TensorView<T> other) const 
        {
            if(rank != other.rank) return false;
            for(size_t i = 0; i < rank; i++)
            {
                if(host_shape[i] != other.host_shape[i] || host_stride[i] != other.host_stride[i]) return false;
            }
            return true;
        }
    
        template <typename... Indices>
        __host__ 
        T& at_host(Indices... indices) {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index_h(idx_array)];
        }        
        template <typename... Indices>
        __host__
        const T& at_host(Indices... indices) const {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index_h(idx_array)];
        }

        template <typename... Indices>
        __device__
        T& at_device(Indices... indices) {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index_d(idx_array)];
        }        
        template <typename... Indices>
        __device__
        const T& at_device(Indices... indices) const {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index_d(idx_array)];
        }

        __host__ __device__
        T& operator[](size_t flat_index) {
            return data[flat_index];
        }

        __host__ __device__
        const T& operator[](size_t flat_index) const {
            return data[flat_index];
        }


        __host__
        size_t compute_flat_index_h(const size_t* indices) const {
            size_t flat = 0;
            for (size_t i = 0; i < rank; ++i) {
                flat += indices[i] * host_stride[i];
            }
            return flat;
        }
        __device__
        size_t compute_flat_index_d(const size_t* indices) const {
            size_t flat = 0;
            for (size_t i = 0; i < rank; ++i) {
                flat += indices[i] * device_stride[i];
            }
            return flat;
        }

        // __host__ __device__
        // void compute_multi_index(size_t flat_index, size_t* indices_out) const {
        //     for (size_t i = 0; i < rank; ++i) {
        //         indices_out[i] = flat_index / device_stride[i];
        //         flat_index %= device_stride[i];
        //     }
        // }
        
        __host__
        size_t size_h() const
        {
            size_t acc = 1;
            for(int i = 0; i < rank; ++i)
                acc *= host_shape[i];
            return acc;
        }
        __device__
        size_t size_d() const
        {
            size_t acc = 1;
            for(int i = 0; i < rank; ++i)
                acc *= device_shape[i];
            return acc;
        }
    private:
        void free_device_metadata() {
            if (device_shape) {cudaFree(device_shape); device_shape = nullptr;};            
            if (device_stride) {cudaFree(device_stride); device_stride = nullptr;}            
        }
    };
}
