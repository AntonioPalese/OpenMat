#include "type_traits/types.cuh"
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include "cuda_defines.cuh"
#include <stdexcept>
#include <cuda_runtime.h>

namespace om 
{

    constexpr int MATMUL_TILE_SIZE = 16;

    template<typename T>
    __global__ void matmul_kernel(
        const T* __restrict__ A,
        const T* __restrict__ B,
        T* __restrict__ C,
        size_t M, size_t K, size_t N,
        size_t strideA0, size_t strideA1,
        size_t strideB0, size_t strideB1,
        size_t strideC0, size_t strideC1)
    {

        __shared__ T tileA[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
        __shared__ T tileB[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];


        int tx = threadIdx.x;
        int ty = threadIdx.y;

      
        size_t row = blockIdx.y * MATMUL_TILE_SIZE + ty;
        size_t col = blockIdx.x * MATMUL_TILE_SIZE + tx;

        T sum = static_cast<T>(0);


        int numTiles = (K + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE;

        for (int t = 0; t < numTiles; ++t) {

            size_t aCol = t * MATMUL_TILE_SIZE + tx;
            tileA[ty][tx] = (row < M && aCol < K) ? A[row * strideA0 + aCol * strideA1] : static_cast<T>(0);


            size_t bRow = t * MATMUL_TILE_SIZE + ty;
            tileB[ty][tx] = (bRow < K && col < N) ? B[bRow * strideB0 + col * strideB1] : static_cast<T>(0);

            __syncthreads();


            #pragma unroll
            for (int k = 0; k < MATMUL_TILE_SIZE; ++k) {
                sum = sum + (tileA[ty][k] * tileB[k][tx]);
            }

            __syncthreads();
        }


        if (row < M && col < N) {
            C[row * strideC0 + col * strideC1] = sum;
        }
    }

    template<typename T>
    void launch_matmul(const TensorView<const T> lhs, const TensorView<const T> rhs, TensorView<T> dst)
    {

        if (lhs.rank != 2 || rhs.rank != 2 || dst.rank != 2) {
            throw std::runtime_error("launch_matmul: all tensors must be 2D matrices");
        }

        size_t M = lhs.shape[0];
        size_t K = lhs.shape[1];
        size_t K2 = rhs.shape[0];
        size_t N = rhs.shape[1];

        if (K != K2) {
            throw std::runtime_error("launch_matmul: inner dimensions must match");
        }

        if (dst.shape[0] != M || dst.shape[1] != N) {
            throw std::runtime_error("launch_matmul: output dimensions mismatch");
        }

 
        dim3 threads(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE);
        dim3 blocks(
            (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
            (M + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE
        );

        matmul_kernel<T><<<blocks, threads>>>(
            lhs.data, rhs.data, dst.data,
            M, K, N,
            lhs.stride[0], lhs.stride[1],
            rhs.stride[0], rhs.stride[1],
            dst.stride[0], dst.stride[1]
        );

        CUDA_CHECK;
        cudaDeviceSynchronize();
    }


    template void launch_matmul<float>(
        const TensorView<const float> lhs, 
        const TensorView<const float> rhs, 
        TensorView<float> dst);

    template void launch_matmul<int>(
        const TensorView<const int> lhs, 
        const TensorView<const int> rhs, 
        TensorView<int> dst);

    template void launch_matmul<double>(
        const TensorView<const double> lhs, 
        const TensorView<const double> rhs, 
        TensorView<double> dst);

    template void launch_matmul<float16_t>(
        const TensorView<const float16_t> lhs, 
        const TensorView<const float16_t> rhs, 
        TensorView<float16_t> dst);
}
