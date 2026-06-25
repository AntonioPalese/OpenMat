#pragma once
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include "cuda_defines.cuh"
#include <stdexcept>
#include <cuda_runtime.h>

namespace om
{
    // Transposes a 2D matrix src (M×N) into dst (N×M).
    template<typename T>
    void launch_transpose(const TensorView<const T> src, TensorView<T> dst, cudaStream_t stream = 0);

    // Permutes axes of src according to `axes` and writes into dst.
    // dst must already have the permuted shape allocated.
    template<typename T>
    void launch_permute(const TensorView<const T> src, TensorView<T> dst,
                        const size_t* axes, size_t rank, cudaStream_t stream = 0);
}
