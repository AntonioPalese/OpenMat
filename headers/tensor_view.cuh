#pragma once
#include <vector>
#include <assert.h>
#include <type_traits>
#include <numeric>

#include "cuda_defines.cuh"
#include "device_tensor_view.cuh"

namespace om {
    template<typename T>
    struct TensorView {
        T* data;
        const size_t* shape;
        const size_t* stride;        
        size_t rank;

        __host__
        bool match(TensorView<T> other) const
        {
            if(rank != other.rank) return false;
            for(size_t i = 0; i < rank; i++)
            {
                if(shape[i] != other.shape[i] || stride[i] != other.stride[i]) return false;
            }
            return true;
        }
    
        template <typename... Indices>
        __host__
        T& operator()(Indices... indices) {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index(idx_array)];
        }        
        template <typename... Indices>
        __host__
        const T& operator()(Indices... indices) const {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");

            constexpr size_t num_indices = sizeof...(Indices);
            assert(num_indices == rank && "Incorrect number of indices for tensor access.");

            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index(idx_array)];
        }

        template<typename U = T,
        typename = std::enable_if_t<!std::is_const<U>::value>>
        __host__
        operator TensorView<const T>() const {
            return TensorView<const T>{data, shape, stride, rank};
        }

        __host__
        T& operator[](size_t flat_index) {
            return data[flat_index];
        }

        __host__
        const T& operator[](size_t flat_index) const {
            return data[flat_index];
        }

        __host__
        size_t compute_flat_index(const size_t* indices) const {
            size_t flat = 0;
            for (size_t i = 0; i < rank; ++i) {
                flat += indices[i] * stride[i];
            }
            return flat;
        }

        __host__
        void compute_multi_index(size_t flat_index, size_t* indices_out) const {
            for (size_t i = 0; i < rank; ++i) {
                indices_out[i] = flat_index / stride[i];
                flat_index %= stride[i];
            }
        }
        
        __host__
        size_t size() const
        {
            size_t acc = 1;
            for(int i = 0; i < rank; ++i)
                acc *= shape[i];
            return acc;
        }

        __host__
        DeviceTensorView<const T> as_device_tw() const
        {
            return DeviceTensorView<const T>(
                data, 
                shape,
                stride,
                rank
            );
        }

        __host__
        DeviceTensorView<T> as_device_tw()
        {
            return DeviceTensorView<T>(
                data, 
                shape,
                stride,
                rank
            );
        }
    };
}
