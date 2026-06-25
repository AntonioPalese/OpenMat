#pragma once
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include <cuda_runtime.h>

namespace om
{
    template<typename T> T launch_reduce_sum(const TensorView<const T> src, cudaStream_t stream = 0);
    template<typename T> T launch_reduce_min(const TensorView<const T> src, cudaStream_t stream = 0);
    template<typename T> T launch_reduce_max(const TensorView<const T> src, cudaStream_t stream = 0);
}
