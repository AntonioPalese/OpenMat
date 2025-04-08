#include <stdexcept>
#include "ops/kernels/fill_gpu.cuh"
#include "cuda_defines.cuh"

namespace om {

    
    template<typename T>
    __global__ void fill_kernel_rank1(TensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if(x < tensor.shape[0])
            tensor(x) = value;
    }
    
    template<typename T>
    __global__ void fill_kernel_rank2(TensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(y < tensor.shape[0] && x < tensor.shape[1])
            tensor(y, x) = value;
    }
    
    template<typename T>
    __global__ void fill_kernel_rank3(TensorView<T> tensor, T value) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(z < tensor.shape[0] && y < tensor.shape[1] && x < tensor.shape[2])
            tensor(z, y, x) = value;
    }

    template<typename T>
    __global__ void fill_kernel_rank4(TensorView<T> tensor, T value) {
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
    __global__ void fill_kernel_nd(TensorView<T> tensor, T value) {
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

    template<typename T>
    void launch_fill(TensorView<T> tensor, T value)
    {
        switch (tensor.rank)
        {
        case 1:
            {
                dim3 threads(16);
                dim3 blocks((tensor.shape[0] + 15) / 16);
                fill_kernel_rank1<<<blocks, threads>>>(tensor, value);
            }
            break;
        case 2:
            {
                dim3 threads(16, 16);
                dim3 blocks((tensor.shape[1] + 15) / 16, (tensor.shape[0] + 15) / 16);
                fill_kernel_rank2<<<blocks, threads>>>(tensor, value);
            }
            break;        
        case 3:
            {
                dim3 threads(8, 8, 8); // solo 512 thread per blocco
                dim3 blocks((tensor.shape[2] + 7) / 8, (tensor.shape[1] + 7) / 8, (tensor.shape[0] + 7) / 8);
                fill_kernel_rank3<<<blocks, threads>>>(tensor, value);
            }
            break;  
        case 4:
            {
                dim3 threads(8, 8, tensor.shape[1] < 8 ? tensor.shape[1] : 8); // evita z troppo grandi
                dim3 blocks(
                    (tensor.shape[3] + threads.x - 1) / threads.x,
                    (tensor.shape[2] + threads.y - 1) / threads.y,
                    tensor.shape[0]
                );                             
                fill_kernel_rank4<<<blocks, threads>>>(tensor, value);
            }
            break;            
        default:
            {
                size_t total_elements = tensor.size();
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                fill_kernel_nd<<<blocks, threads>>>(tensor, value);
            }
            break;
        }
        CUDA_CHECK;
        cudaDeviceSynchronize();
    }

    // Explicit instantiations
    template void launch_fill<float>(TensorView<float>, float);
    template void launch_fill<int>(TensorView<int>, int);
    template void launch_fill<char>(TensorView<char>, char);
}
