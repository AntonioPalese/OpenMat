#pragma once
#include "tensor_view.cuh"
#include "type_traits/types.cuh"
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <assert.h>

namespace om
{
    template<typename T>
    void matmul_cpu(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst) {
        static_assert(is_extended_arithmetic<T>::value, "matmul requires an arithmetic type");

        if (lhs.rank != 2 || rhs.rank != 2 || dst.rank != 2) {
            throw std::runtime_error("matmul_cpu: all tensors must be 2D matrices");
        }

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

        // Every Tensor buffer reaching here is canonical row-major contiguous
        // (the library never hands matmul an aliasing/strided view — see
        // CLAUDE.md's Shape ops section), so the inner dimension always has
        // unit stride. That lets the hot loop walk raw pointers instead of
        // paying compute_flat_index's multiply-add per element access.
        assert(lhs.stride[1] == 1 && rhs.stride[1] == 1 && dst.stride[1] == 1 &&
               "matmul_cpu: expected contiguous row-major operands");

        const T* A = lhs.data;
        const T* B = rhs.data;
        T* C = dst.data;
        const size_t lda = lhs.stride[0];
        const size_t ldb = rhs.stride[0];
        const size_t ldc = dst.stride[0];

        std::fill(C, C + M * ldc, T{0});

        // ikj order + L2 tiling: the original ijk loop indexed rhs(k, j) down
        // a column, one cache-line miss per k for a single output element.
        // Walking k in the middle instead makes both the rhs row and the dst
        // row swept contiguously (unit stride, vectorizable) in the innermost
        // j loop; blocking i/k/j keeps each panel resident in L2 across the
        // accumulation. Rows are independent, so they're split across threads.
        constexpr size_t TILE = 128;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            const T* Ai = A + i * lda;
            T* Ci = C + i * ldc;
            for (size_t k0 = 0; k0 < K; k0 += TILE) {
                const size_t kmax = std::min(k0 + TILE, K);
                for (size_t j0 = 0; j0 < N; j0 += TILE) {
                    const size_t jmax = std::min(j0 + TILE, N);
                    for (size_t k = k0; k < kmax; ++k) {
                        const T a = Ai[k];
                        const T* Bk = B + k * ldb;
                        for (size_t j = j0; j < jmax; ++j) {
                            Ci[j] = Ci[j] + a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}
