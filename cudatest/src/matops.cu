#include "matops.h"
#include "cudautils.h"
#include <stdio.h>
#include <cuda.h>
#include <cassert>

__device__
float* cudaAt(Matrix M, int r, int c)
{
    return M.data + c+ r * M.cols;
}

__global__ void _add(Matrix A, Matrix B, Matrix C)
{
    //printf("kernel _add\n");
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    *cudaAt(C, y, x) = *cudaAt(A, y, x) + *cudaAt(B, y, x);
}

__global__ void _mul(Matrix A, Matrix B, Matrix C)
{
    //printf("kernel _add\n");
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    float acc = 0;
    for(int i = 0; i < A.cols; i++)
    {
        acc += *cudaAt(A, y, i) * *cudaAt(B, i, x);
    }
    *cudaAt(C, y, x) = acc;
}

__host__
void cuda_add(Matrix A, Matrix B, Matrix C)
{    
    Matrix gpuA, gpuB, gpuC;

    to_device(&gpuA, &A);
    to_device(&gpuB, &B);
    to_device(&gpuC, &C);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // x, y, z
    dim3 gridDim(A.cols/BLOCK_SIZE, A.rows/BLOCK_SIZE);
    _add<<<gridDim, blockDim>>>(gpuA, gpuB, gpuC);

    cudaError_t err = cudaGetLastError();
    print_cuda_err(err, "_add kernel");
    err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");

    to_host(&C, &gpuC);

    err = cudaFree(gpuA.data);
    print_cuda_err(err, "cudaFree");
    err = cudaFree(gpuB.data);
    print_cuda_err(err, "cudaFree");
    err = cudaFree(gpuC.data);
    print_cuda_err(err, "cudaFree");
}

__host__
void cuda_mul(Matrix A, Matrix B, Matrix C)
{    
    Matrix gpuA, gpuB, gpuC;

    to_device(&gpuA, &A);
    to_device(&gpuB, &B);
    to_device(&gpuC, &C);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // x, y, z
    dim3 gridDim(A.cols/BLOCK_SIZE, A.rows/BLOCK_SIZE);
    _mul<<<gridDim, blockDim>>>(gpuA, gpuB, gpuC);

    cudaError_t err = cudaGetLastError();
    print_cuda_err(err, "_add kernel");
    err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");

    to_host(&C, &gpuC);

    err = cudaFree(gpuA.data);
    print_cuda_err(err, "cudaFree");
    err = cudaFree(gpuB.data);
    print_cuda_err(err, "cudaFree");
    err = cudaFree(gpuC.data);
    print_cuda_err(err, "cudaFree");
}

__host__
void host_add(Matrix A, Matrix B, Matrix C)
{
    for(int i = 0; i < A.rows; i++)
    {
        for(int j = 0; j < A.rows; j++)
        {
            *at(C, i, j) = *at(A, i, j) + *at(B, i, j);
        }
    }
}

__host__
void host_mul(Matrix A, Matrix B, Matrix C)
{
    for(int i = 0; i < A.rows; i++)
    {
        for(int j = 0; j < B.cols; j++)
        {
            int acc = 0;
            for(int k = 0; k < A.cols; k++)
            {
                acc += *at(A, i, k) * *at(B, k, j); 
            }
            *at(C, i, j) = acc;
        }
    }
}

void add(Matrix A, Matrix B, Matrix C)
{
    assert(A.cols == B.rows);
    if(A.cuda && B.cuda && C.cuda)
        cuda_add(A, B, C);
    else
        host_add(A, B, C);    
}

void mul(Matrix A, Matrix B, Matrix C)
{
    assert(A.cols == B.rows);
    if(A.cuda && B.cuda && C.cuda)
        cuda_mul(A, B, C);
    else
        host_mul(A, B, C);    
}