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

__host__
void cuda_add(Matrix A, Matrix B, Matrix C)
{    
    Matrix gpuA, gpuB, gpuC;

    to_device(&gpuA, &A);
    to_device(&gpuB, &B);
    to_device(&gpuC, &C);

    dim3 thDim(gpuA.cols,gpuA.rows); // x, y, z
    _add<<<1, thDim>>>(gpuA, gpuB, gpuC);

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

}

void add(Matrix A, Matrix B, Matrix C)
{
    assert(A.cols == B.rows);
    if(A.cuda && B.cuda)
        cuda_add(A, B, C);
    else
        host_add(A, B, C);    
}