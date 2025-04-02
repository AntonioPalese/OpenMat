#include <stdexcept>
#include "ops/kernels/fill_gpu.cuh"
#include "cuda_defines.cuh"

namespace om {

    template<typename T>
    __global__ void fill_kernel(MatView<T> mat, T value) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;

        if (r < mat.rows && c < mat.cols)
            mat(r, c) = value;
    }

    template<typename T>
    void launch_fill(MatView<T> mat, T value) {
        dim3 threads(16, 16);
        dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
        fill_kernel<<<blocks, threads>>>(mat, value);
        CUDA_CHECK;
        cudaDeviceSynchronize();
    }

    // Explicit instantiations
    template void launch_fill<float>(MatView<float>, float);
    template void launch_fill<int>(MatView<int>, int);
    template void launch_fill<char>(MatView<char>, char);
}
