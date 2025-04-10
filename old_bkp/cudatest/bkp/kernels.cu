#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>


// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    C[i] = A[i] + B[i];
}

void m_set(float* src, float val, int size)
{
    for(int i = 0; i < size; i++)
    {
        src[i] = val;
    }
}

void print_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        printf("[CUDA error] : %s\n", cudaGetErrorString(err));
}

int main()
{
    const int N = 1024*10000;
    // Kernel invocation with N threads
    float* host_A = (float*)malloc(N*sizeof(float));
    float* host_B = (float*)malloc(N*sizeof(float));
    float* host_C = (float*)malloc(N*sizeof(float));

    cudaSetDevice(0);

    float* A, *B, *C;
    cudaError_t err;
    err = cudaMalloc(&A, N*sizeof(float));
    print_cuda_error(err);
    err = cudaMalloc(&B, N*sizeof(float));
    print_cuda_error(err);
    err = cudaMalloc(&C, N*sizeof(float));
    print_cuda_error(err);
       
       
    m_set(host_A, 0.0f, N);
    m_set(host_B, 11.0f, N);
    m_set(host_C, 0.0f, N);

    err = cudaMemcpy(A, host_A, N*sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_error(err);
    err = cudaMemcpy(B, host_B, N*sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_error(err);
    err = cudaMemcpy(C, host_C, N*sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_error(err);

    auto start = std::chrono::high_resolution_clock::now();
    VecAdd<<<2, N/2>>>(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    err = cudaMemcpy(host_C, C, N*sizeof(float), cudaMemcpyDeviceToHost);
    print_cuda_error(err);

    std::cout << "function elapsed time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " micosecs" << std::endl;

    // for(int i = 0;i<N; i++)
    // {
    //     printf("C[%d] = %f\n", i, host_C[i]);
    // }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    free(host_A);
    free(host_B);
    free(host_C);

}