#pragma once
#include "tensor_view.cuh"

namespace om 
{
    // Forward declaration of launch_matmul
    // Implementation is in matmul_gpu.cu to avoid CUDA code in .cpp files
    
    /**
     * @brief Launch matmul kernel: C = A × B
     * 
     * @tparam T Arithmetic type (float, int, double)
     * @param lhs Left matrix A (M × K)
     * @param rhs Right matrix B (K × N)
     * @param dst Output matrix C (M × N)
     */
    template<typename T>
    void launch_matmul(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst, cudaStream_t stream = 0);
}
