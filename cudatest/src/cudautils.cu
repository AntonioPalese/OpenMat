#include "cudautils.h"
#include <stdio.h>

__global__ void _print(float* A, size_t r, size_t c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", A[j+i*c]);
        }
        printf("\n");
    }
}

void print_cuda_err(cudaError_t err, const char* tag)
{
    if(err != cudaSuccess)
        printf("[CUDA ERROR (%s)] :  %s\n", tag, cudaGetErrorString(err));
}

void cuda_print(float* mat, size_t r, size_t c)
{
    _print<<<1, 1>>>(mat, r, c);
}

void print(float* mat, size_t r, size_t c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", mat[j+c*i]);
        }
        printf("\n");
    }
}

void to_device(Matrix* dst, const Matrix* src)
{
    dst->cols = src->cols;
    dst->rows = src->rows;
    dst->cuda = true;
    dst->stride = src->stride;

    cudaDeviceSynchronize();
    size_t size = src->rows*src->cols*sizeof(float);
    cudaError_t err = cudaMalloc((void**)&dst->data, size);
    print_cuda_err(err, "cudaMalloc");
    err = cudaMemcpy(dst->data, src->data, size, cudaMemcpyHostToDevice);
    print_cuda_err(err, "cudaMemcpy");
}

void to_host(Matrix* dst, const Matrix* src)
{
    cudaDeviceSynchronize();
    size_t size = src->rows*src->cols*sizeof(float);
    cudaError_t err = cudaMemcpy(dst->data, src->data, size, cudaMemcpyDeviceToHost);
    print_cuda_err(err, "cudaMemcpy");
}