#include <stdio.h>
#include <string>

#define CUDA_CHECK(op) cuda_check(op, #op);
void cuda_check(cudaError_t err, const char* desc)
{
    if(err != cudaError::cudaSuccess)
    {
        printf("[CUDA ERROR ( %s )] : %s\n", desc, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// -------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------


__global__ void kernelTest()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("idx : %d\n", idx);
}

int main(void)
{
    size_t n = pow(2, 28);
    size_t size = n*sizeof(float);

    float* h_a, * h_b;
    float* d_a;

    

    CUDA_CHECK(cudaMallocHost((void**)&h_a, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    //h_a = (float*) malloc(size);
    //h_b = (float*) malloc(size);    

    memset(h_a, 0xffffffff, size);

    /////////    

    cudaStream_t streams[2];
    for(int i = 0; i < 2; i++)
    {
         CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    for(int i = 0; i < 2; i++)
    {
        int half_size = size/2;
        int half_elements = n/2;
        CUDA_CHECK(cudaMemcpyAsync(d_a + i * half_elements, h_a + i * half_elements, half_size, cudaMemcpyHostToDevice, streams[i]));  

        dim3 blockDim(1024);
        dim3 gridDim(n / 1024);
        kernelTest<<<gridDim, blockDim, 0, streams[i]>>>();
        CUDA_CHECK(cudaGetLastError());  

        CUDA_CHECK(cudaMemcpyAsync(h_b + i * half_elements, d_a + i * half_elements, half_size, cudaMemcpyDeviceToHost, streams[i]));
        //CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    for(int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }


    /////////

    // CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // kernelTest<<<1, n>>>();
    // CUDA_CHECK(cudaGetLastError());

    // CUDA_CHECK(cudaMemcpy(h_b, d_a, size, cudaMemcpyDeviceToHost ));

    CUDA_CHECK(cudaFree(d_a));
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    // free(h_a);
    // free(h_b);

}