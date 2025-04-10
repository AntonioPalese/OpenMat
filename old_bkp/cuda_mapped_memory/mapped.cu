#include <stdio.h>
#include <string>
#include <stdlib.h>

//#define STRCAT(dst, src) strcat(dst, src)
#define CUDA_CHECK(op) cuda_check(op,#op)
void cuda_check(cudaError_t err, const char* desc)
{
    if(err != cudaError::cudaSuccess)
    {
        printf("[CUDA ERROR ( %s )] : %s\n", desc, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void hostMempryAccessingKernel(float *dptr)
{
    *dptr = 1.0f;
    printf("dptr in kernel : %f\n", *dptr);
}

int main(void)
{
    size_t n = 1;
    size_t size = n * sizeof(float);


    float* h_a;
    
    CUDA_CHECK(cudaHostAlloc(&h_a, size, cudaHostAllocMapped));
    float* dprt;
    CUDA_CHECK(cudaHostGetDevicePointer(&dprt, h_a, 0));
    hostMempryAccessingKernel<<<1, 1>>>(dprt);
    CUDA_CHECK(cudaGetLastError());

    // the kernel is executed asynch, even before the printf buffer could be loaded
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("value of the pointer accessed by host after kernal execution : %f\n", *h_a);
}
