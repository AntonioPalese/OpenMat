#include <stdexcept>
#include "ops/kernels/fill_gpu.cuh"
#include "ops/kernels/contiguous.cuh"
#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

namespace om {

    
    template<typename T>
    __global__ void fill_kernel_rank1(DeviceTensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if(x < tensor.shape[0])
            tensor(x) = value;
    }
    
    template<typename T>
    __global__ void fill_kernel_rank2(DeviceTensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(y < tensor.shape[0] && x < tensor.shape[1])
            tensor(y, x) = value;
    }
    
    template<typename T>
    __global__ void fill_kernel_rank3(DeviceTensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(z < tensor.shape[0] && y < tensor.shape[1] && x < tensor.shape[2])
            tensor(z, y, x) = value;
    }

    template<typename T>
    __global__ void fill_kernel_rank4(DeviceTensorView<T> tensor, T value) {
        size_t w = threadIdx.x + blockIdx.x * blockDim.x;
        size_t h = threadIdx.y + blockIdx.y * blockDim.y;
        size_t n = blockIdx.z; // N dimension
    
        for (size_t c = threadIdx.z; c < tensor.shape[1]; c += blockDim.z) {
            if (n < tensor.shape[0] &&
                c < tensor.shape[1] &&
                h < tensor.shape[2] &&
                w < tensor.shape[3]) {
                    tensor(n, c, h, w) = value;
            }
        }
    }
    
    template<typename T>
    __global__ void fill_kernel_nd(DeviceTensorView<T> tensor, T value) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_elements = tensor.size();
    
        if (idx >= total_elements) return;
    
        size_t offset = 0;
        size_t tmp = idx;
        for (size_t d = 0; d < tensor.rank; ++d) {
            size_t coord = tmp % tensor.shape[d];
            offset += coord * tensor.stride[d];
            tmp /= tensor.shape[d];
        }
    
        tensor[offset] = value;
    }

    // Rank-specialized launch. Each rank maps tensor axes onto grid axes, and
    // gridDim.y/z stop at 65535: when the specialized grid does not fit, fall
    // back to the flat _nd kernel (gridDim.x only) instead of failing the
    // launch with "invalid configuration argument".
    // Rank-1 blocks are 256 threads: a 16-thread block fills half a warp and
    // leaves the other 16 lanes idle for the whole launch, which made a vector
    // slower than the very same buffer viewed as a matrix.
    template<typename T>
    void launch_fill(TensorView<T> tensor, T value, cudaStream_t stream)
    {
        const char* om_kernel = nullptr;
        bool om_use_nd = true;
        // Contiguous fast path — see headers/ops/kernels/contiguous.cuh.
        if (tensor.is_contiguous())
        {
            om_kernel = detail::launch_contiguous_fill<T>(tensor.data, value, tensor.size(), stream);
            om_use_nd = (om_kernel == nullptr);
        }
        if (om_use_nd)
        switch (tensor.rank)
        {
        case 1:
            {
                dim3 threads(256);
                const size_t gx = (tensor.shape[0] + 255) / 256;
                if (detail::grid_fits(gx, 1, 1))
                {
                    dim3 blocks(static_cast<unsigned int>(gx));
                    om_kernel = "fill_kernel_rank1";
                    om_use_nd = false;
                    fill_kernel_rank1<<<blocks, threads, 0, stream>>>(tensor.as_device_tw(), value);
                }
            }
            break;
        case 2:
            {
                dim3 threads(16, 16);
                const size_t gx = (tensor.shape[1] + 15) / 16;
                const size_t gy = (tensor.shape[0] + 15) / 16;
                if (detail::grid_fits(gx, gy, 1))
                {
                    dim3 blocks(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy));
                    om_kernel = "fill_kernel_rank2";
                    om_use_nd = false;
                    fill_kernel_rank2<<<blocks, threads, 0, stream>>>(tensor.as_device_tw(), value);
                }
            }
            break;
        case 3:
            {
                dim3 threads(8, 8, 8);
                const size_t gx = (tensor.shape[2] + 7) / 8;
                const size_t gy = (tensor.shape[1] + 7) / 8;
                const size_t gz = (tensor.shape[0] + 7) / 8;
                if (detail::grid_fits(gx, gy, gz))
                {
                    dim3 blocks(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(gz));
                    om_kernel = "fill_kernel_rank3";
                    om_use_nd = false;
                    fill_kernel_rank3<<<blocks, threads, 0, stream>>>(tensor.as_device_tw(), value);
                }
            }
            break;
        case 4:
            {
                dim3 threads(8, 8, tensor.shape[1] < 8 ? tensor.shape[1] : 8);
                const size_t gx = (tensor.shape[3] + threads.x - 1) / threads.x;
                const size_t gy = (tensor.shape[2] + threads.y - 1) / threads.y;
                const size_t gz = tensor.shape[0];
                if (detail::grid_fits(gx, gy, gz))
                {
                    dim3 blocks(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(gz));
                    om_kernel = "fill_kernel_rank4";
                    om_use_nd = false;
                    fill_kernel_rank4<<<blocks, threads, 0, stream>>>(tensor.as_device_tw(), value);
                }
            }
            break;
        default:
            break;
        }
        if (om_use_nd)
        {
            size_t total_elements = tensor.size();
            dim3 threads(256);
            dim3 blocks(static_cast<unsigned int>((total_elements + threads.x - 1) / threads.x));
            om_kernel = "fill_kernel_nd";
            fill_kernel_nd<<<blocks, threads, 0, stream>>>(tensor.as_device_tw(), value);
        }
        CUDA_CHECK_LAUNCH(om_kernel, stream);
        if (stream == nullptr) cudaDeviceSynchronize();
    }

    // Explicit instantiations
    template void launch_fill<float>    (TensorView<float>,     float,     cudaStream_t);
    template void launch_fill<int>      (TensorView<int>,       int,       cudaStream_t);
    template void launch_fill<char>     (TensorView<char>,      char,      cudaStream_t);
    template void launch_fill<float16_t>(TensorView<float16_t>, float16_t, cudaStream_t);
}
