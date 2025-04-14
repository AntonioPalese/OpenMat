#include <stdexcept>
#include "ops/kernels/fused_op.cuh"
#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

namespace om {

    
    template <typename T, typename Op>
    __global__ void apply_op_rank1(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if(x < tensor.shape[0])
            dst(x) = op(src[idx]);
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_rank2(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(y < tensor.shape[0] && x < tensor.shape[1])
            dst(y, x) = op(src(y, x));
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_rank3(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(z < tensor.shape[0] && y < tensor.shape[1] && x < tensor.shape[2])
            dst(z, y, x) =  op(src(z, y, x));
    }

    template <typename T, typename Op>
    __global__ void apply_op_rank4(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
        size_t w = threadIdx.x + blockIdx.x * blockDim.x;
        size_t h = threadIdx.y + blockIdx.y * blockDim.y;
        size_t n = blockIdx.z; // N dimension
    
        for (size_t c = threadIdx.z; c < tensor.shape[1]; c += blockDim.z) {
            if (n < tensor.shape[0] &&
                c < tensor.shape[1] &&
                h < tensor.shape[2] &&
                w < tensor.shape[3]) {
                    dst(n, c, h, w) =  op(src(n, c, h, w));
            }
        }
    }
    
    template <typename T, typename Op>
    __global__ void apply_op_nd(const DeviceTensorView<const T> src, DeviceTensorView<T> dst, Op op) {
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
    
        dst[offset] = op(src[offset]);
    }

    template <typename T, typename Op>
    void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op)
    {
        switch (tensor.rank)
        {
        case 1:
            {
                dim3 threads(16);
                dim3 blocks((tensor.shape[0] + 15) / 16);
                apply_op_rank1<<<blocks, threads>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        case 2:
            {
                dim3 threads(16, 16);
                dim3 blocks((tensor.shape[1] + 15) / 16, (tensor.shape[0] + 15) / 16);
                apply_op_rank2<<<blocks, threads>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;        
        case 3:
            {
                dim3 threads(8, 8, 8); // solo 512 thread per blocco
                dim3 blocks((tensor.shape[2] + 7) / 8, (tensor.shape[1] + 7) / 8, (tensor.shape[0] + 7) / 8);
                apply_op_rank3<<<blocks, threads>>>(src.as_device_tw(), dst.as_device_tw(), op);
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
                apply_op_rank4<<<blocks, threads>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;            
        default:
            {
                size_t total_elements = tensor.size();
                dim3 threads(256);
                dim3 blocks((total_elements + threads.x - 1) / threads.x);
                apply_op_nd<<<blocks, threads>>>(src.as_device_tw(), dst.as_device_tw(), op);
            }
            break;
        }
        CUDA_CHECK;
        cudaDeviceSynchronize();
    }

    // Explicit instantiations
    template void launch_apply_op<float>(const TensorView<const float> src, TensorView<float> dst, Add<float> op);
    template void launch_apply_op<int>(const TensorView<const int> src, TensorView<int> dst, Add<int> op);
    template void launch_apply_op<char>(const TensorView<const char> src, TensorView<char> dst, Add<char> op);
    template void launch_apply_op<float16_t>(const TensorView<const float16_t> src, TensorView<float16_t> dst, Add<float16_t> op);
}
