#pragma once
#include <vector>
#include <type_traits>

namespace om {
    template<typename T>
    struct TensorView {
        T* data;
        const size_t* shape;    // raw pointers: can be passed to device
        const size_t* strides;
        size_t rank;
    
        template <typename... Indices>
        __host__ __device__
        T& operator()(Indices... indices) {
            static_assert(sizeof...(Indices) > 0, "Must provide at least one index.");
            size_t idx_array[] = { static_cast<size_t>(indices)... };
            return data[compute_flat_index(idx_array)];
        }

        __host__ __device__
        size_t compute_flat_index(const size_t* indices) const {
            size_t flat = 0;
            for (size_t i = 0; i < rank; ++i) {
                flat += indices[i] * strides[i];
            }
            return flat;
        }
    };
}
