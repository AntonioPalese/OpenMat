#include "ops/kernels/reduce_gpu.cuh"
#include "cuda_defines.cuh"
#include "type_traits/types.cuh"
#include <stdexcept>
#include <limits>
#include <cuda_runtime.h>

namespace om {

    // -------------------------------------------------------------------------
    // Warp-level primitives
    // -------------------------------------------------------------------------

    template<typename T>
    __device__ __forceinline__ T warp_reduce_sum(T val) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val = val + __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    template<typename T>
    __device__ __forceinline__ T warp_reduce_min(T val) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, val, offset);
            if (other < val) val = other;
        }
        return val;
    }

    template<typename T>
    __device__ __forceinline__ T warp_reduce_max(T val) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        return val;
    }

    // float16_t is not directly supported by __shfl_down_sync — promote to float
    template<>
    __device__ __forceinline__ float16_t warp_reduce_sum<float16_t>(float16_t val) {
        float f = float(val);
        for (int offset = 16; offset > 0; offset >>= 1)
            f = f + __shfl_down_sync(0xffffffff, f, offset);
        return float16_t(f);
    }

    template<>
    __device__ __forceinline__ float16_t warp_reduce_min<float16_t>(float16_t val) {
        float f = float(val);
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, f, offset);
            if (other < f) f = other;
        }
        return float16_t(f);
    }

    template<>
    __device__ __forceinline__ float16_t warp_reduce_max<float16_t>(float16_t val) {
        float f = float(val);
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, f, offset);
            if (other > f) f = other;
        }
        return float16_t(f);
    }

    // -------------------------------------------------------------------------
    // Block-level reduction kernels — write one partial per block to out[]
    // -------------------------------------------------------------------------

    template<typename T>
    __global__ void reduce_sum_kernel(const T* __restrict__ data, T* __restrict__ out, size_t n) {
        __shared__ unsigned char smem_raw[32 * sizeof(T)];
        T* smem = reinterpret_cast<T*>(smem_raw);

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = static_cast<T>(0);
        if (idx < n) val = data[idx];

        val = warp_reduce_sum(val);

        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = val;
        __syncthreads();

        if (warp == 0) {
            val = (lane < (blockDim.x >> 5)) ? smem[lane] : static_cast<T>(0);
            val = warp_reduce_sum(val);
            if (lane == 0) out[blockIdx.x] = val;
        }
    }

    template<typename T>
    __global__ void reduce_min_kernel(const T* __restrict__ data, T* __restrict__ out, size_t n, T identity) {
        __shared__ unsigned char smem_raw[32 * sizeof(T)];
        T* smem = reinterpret_cast<T*>(smem_raw);

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = identity;
        if (idx < n) val = data[idx];

        val = warp_reduce_min(val);

        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = val;
        __syncthreads();

        if (warp == 0) {
            val = (lane < (blockDim.x >> 5)) ? smem[lane] : identity;
            val = warp_reduce_min(val);
            if (lane == 0) out[blockIdx.x] = val;
        }
    }

    template<typename T>
    __global__ void reduce_max_kernel(const T* __restrict__ data, T* __restrict__ out, size_t n, T identity) {
        __shared__ unsigned char smem_raw[32 * sizeof(T)];
        T* smem = reinterpret_cast<T*>(smem_raw);

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        T val = identity;
        if (idx < n) val = data[idx];

        val = warp_reduce_max(val);

        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = val;
        __syncthreads();

        if (warp == 0) {
            val = (lane < (blockDim.x >> 5)) ? smem[lane] : identity;
            val = warp_reduce_max(val);
            if (lane == 0) out[blockIdx.x] = val;
        }
    }

    // -------------------------------------------------------------------------
    // Two-pass launch helpers
    // -------------------------------------------------------------------------

    constexpr int REDUCE_BLOCK = 256;

    template<typename T>
    T launch_reduce_sum(const TensorView<const T> src, cudaStream_t stream)
    {
        size_t n = src.size();
        if (n == 0) return static_cast<T>(0);

        int blocks = static_cast<int>((n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);

        T* d_partial = nullptr;
        CUDA_CALL(cudaMalloc(&d_partial, blocks * sizeof(T)));

        reduce_sum_kernel<<<blocks, REDUCE_BLOCK, 0, stream>>>(src.data, d_partial, n);
        CUDA_CHECK;

        // Must sync before D→H copy; use stream-specific sync when a stream is provided
        if (stream != nullptr)
            cudaStreamSynchronize(stream);
        else
            cudaDeviceSynchronize();

        std::vector<T> h(blocks);
        CUDA_CALL(cudaMemcpy(h.data(), d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_partial));

        T result = static_cast<T>(0);
        for (T v : h) result = result + v;
        return result;
    }

    template<typename T>
    T launch_reduce_min(const TensorView<const T> src, cudaStream_t stream)
    {
        size_t n = src.size();
        if (n == 0) throw std::invalid_argument("reduce_min: empty tensor");

        T identity;
        if constexpr (std::is_floating_point_v<T>)
            identity = std::numeric_limits<T>::infinity();
        else
            identity = std::numeric_limits<T>::max();

        int blocks = static_cast<int>((n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);

        T* d_partial = nullptr;
        CUDA_CALL(cudaMalloc(&d_partial, blocks * sizeof(T)));

        reduce_min_kernel<<<blocks, REDUCE_BLOCK, 0, stream>>>(src.data, d_partial, n, identity);
        CUDA_CHECK;

        if (stream != nullptr)
            cudaStreamSynchronize(stream);
        else
            cudaDeviceSynchronize();

        std::vector<T> h(blocks);
        CUDA_CALL(cudaMemcpy(h.data(), d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_partial));

        T result = identity;
        for (T v : h) if (v < result) result = v;
        return result;
    }

    template<typename T>
    T launch_reduce_max(const TensorView<const T> src, cudaStream_t stream)
    {
        size_t n = src.size();
        if (n == 0) throw std::invalid_argument("reduce_max: empty tensor");

        T identity;
        if constexpr (std::is_floating_point_v<T>)
            identity = -std::numeric_limits<T>::infinity();
        else
            identity = std::numeric_limits<T>::lowest();

        int blocks = static_cast<int>((n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);

        T* d_partial = nullptr;
        CUDA_CALL(cudaMalloc(&d_partial, blocks * sizeof(T)));

        reduce_max_kernel<<<blocks, REDUCE_BLOCK, 0, stream>>>(src.data, d_partial, n, identity);
        CUDA_CHECK;

        if (stream != nullptr)
            cudaStreamSynchronize(stream);
        else
            cudaDeviceSynchronize();

        std::vector<T> h(blocks);
        CUDA_CALL(cudaMemcpy(h.data(), d_partial, blocks * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_partial));

        T result = identity;
        for (T v : h) if (v > result) result = v;
        return result;
    }

    // float16_t: __shfl needs float promotion (done in warp_reduce specializations),
    // but numeric_limits<float16_t> may not exist — specialize the launchers.
    template<>
    float16_t launch_reduce_min<float16_t>(const TensorView<const float16_t> src, cudaStream_t stream)
    {
        size_t n = src.size();
        if (n == 0) throw std::invalid_argument("reduce_min: empty tensor");

        float16_t identity = float16_t(65504.0f);

        int blocks = static_cast<int>((n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
        float16_t* d_partial = nullptr;
        CUDA_CALL(cudaMalloc(&d_partial, blocks * sizeof(float16_t)));

        reduce_min_kernel<<<blocks, REDUCE_BLOCK, 0, stream>>>(src.data, d_partial, n, identity);
        CUDA_CHECK;

        if (stream != nullptr) cudaStreamSynchronize(stream);
        else                   cudaDeviceSynchronize();

        std::vector<float16_t> h(blocks);
        CUDA_CALL(cudaMemcpy(h.data(), d_partial, blocks * sizeof(float16_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_partial));

        float16_t result = identity;
        for (float16_t v : h) if (float(v) < float(result)) result = v;
        return result;
    }

    template<>
    float16_t launch_reduce_max<float16_t>(const TensorView<const float16_t> src, cudaStream_t stream)
    {
        size_t n = src.size();
        if (n == 0) throw std::invalid_argument("reduce_max: empty tensor");

        float16_t identity = float16_t(-65504.0f);

        int blocks = static_cast<int>((n + REDUCE_BLOCK - 1) / REDUCE_BLOCK);
        float16_t* d_partial = nullptr;
        CUDA_CALL(cudaMalloc(&d_partial, blocks * sizeof(float16_t)));

        reduce_max_kernel<<<blocks, REDUCE_BLOCK, 0, stream>>>(src.data, d_partial, n, identity);
        CUDA_CHECK;

        if (stream != nullptr) cudaStreamSynchronize(stream);
        else                   cudaDeviceSynchronize();

        std::vector<float16_t> h(blocks);
        CUDA_CALL(cudaMemcpy(h.data(), d_partial, blocks * sizeof(float16_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(d_partial));

        float16_t result = identity;
        for (float16_t v : h) if (float(v) > float(result)) result = v;
        return result;
    }

    // Explicit instantiations
    template float     launch_reduce_sum<float>    (const TensorView<const float>,     cudaStream_t);
    template int       launch_reduce_sum<int>      (const TensorView<const int>,       cudaStream_t);
    template char      launch_reduce_sum<char>     (const TensorView<const char>,      cudaStream_t);
    template float16_t launch_reduce_sum<float16_t>(const TensorView<const float16_t>, cudaStream_t);

    template float     launch_reduce_min<float>    (const TensorView<const float>,     cudaStream_t);
    template int       launch_reduce_min<int>      (const TensorView<const int>,       cudaStream_t);
    template char      launch_reduce_min<char>     (const TensorView<const char>,      cudaStream_t);

    template float     launch_reduce_max<float>    (const TensorView<const float>,     cudaStream_t);
    template int       launch_reduce_max<int>      (const TensorView<const int>,       cudaStream_t);
    template char      launch_reduce_max<char>     (const TensorView<const char>,      cudaStream_t);
}
