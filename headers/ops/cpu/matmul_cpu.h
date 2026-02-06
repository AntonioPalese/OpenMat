#pragma once
#include "tensor_view.cuh"
#include <type_traits>
#include <stdexcept>

namespace om 
{
    /**
     * @brief CPU matrix multiplication: C = A × B
     * 
     * @tparam T Arithmetic type (float, double, int, etc.)
     * @param lhs Left-hand side matrix (M × K)
     * @param rhs Right-hand side matrix (K × N)  
     * @param dst Output matrix (M × N)
     * 
     * @throws std::runtime_error if dimensions are incompatible
     */
    template<typename T>
    void matmul_cpu(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {
        static_assert(std::is_arithmetic_v<T>, "matmul requires an arithmetic type");

        // Validate ranks
        if (lhs.rank != 2 || rhs.rank != 2 || dst.rank != 2) {
            throw std::runtime_error("matmul_cpu: all tensors must be 2D matrices");
        }

        // A is M×K, B is K×N, C is M×N
        size_t M = lhs.shape[0];
        size_t K = lhs.shape[1];
        size_t K2 = rhs.shape[0];
        size_t N = rhs.shape[1];

        if (K != K2) {
            throw std::runtime_error("matmul_cpu: inner dimensions must match (A.cols != B.rows)");
        }

        if (dst.shape[0] != M || dst.shape[1] != N) {
            throw std::runtime_error("matmul_cpu: output dimensions must be (M × N)");
        }

        // Triple nested loop: C[i,j] = Σ A[i,k] * B[k,j]
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T{0};
                for (size_t k = 0; k < K; ++k) {
                    sum += lhs(i, k) * rhs(k, j);
                }
                dst(i, j) = sum;
            }
        }
    }
}
