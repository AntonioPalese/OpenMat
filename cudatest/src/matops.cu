#include "matops.h"
#include "cudautils.h"
#include <stdio.h>
#include <cuda.h>
#include <cassert>
#include <chrono>

__device__
float* cudaAt(Matrix M, int r, int c)
{
    return M.data + c + r * M.stride;
}

__device__ Matrix subMatrix(Matrix M, int r, int c)
{
    Matrix SubM;
    SubM.rows = BLOCK_SIZE;
    SubM.cols = BLOCK_SIZE;
    SubM.stride = M.stride;
    SubM.data = M.data + SubM.stride*r*BLOCK_SIZE + c*BLOCK_SIZE;

    return SubM;
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

__global__ void _shared_memory_mul(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = subMatrix(C, blockRow, blockCol);
    float acc = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int i = 0; i < A.cols / BLOCK_SIZE; i++)
    {
        Matrix Asub = subMatrix(A, blockRow, i);
        Matrix Bsub = subMatrix(B, i, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = *cudaAt(Asub, row, col);
        Bs[row][col] = *cudaAt(Bsub, row, col);

        __syncthreads();

        for(int e = 0; e < BLOCK_SIZE; e++)
        {
            acc+=As[row][e]*Bs[e][col];            
        }         
        __syncthreads();
    }

    *cudaAt(Csub, row, col) = acc;
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

auto get_ns()
{
    return std::chrono::steady_clock::now();
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

    auto st = get_ns();
    _shared_memory_mul<<<gridDim, blockDim>>>(gpuA, gpuB, gpuC);
    //_mul<<<gridDim, blockDim>>>(gpuA, gpuB, gpuC);
    cudaError_t err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");
    auto et = get_ns();

    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(et - st).count();
    
    double n = A.rows*A.cols;
    double flops = n*n*2*n;
    double gflops = flops * 1e-9;

    double elapsed_s = elapsed_ns * 1e-9;
    printf( "elapsed : %f s\n",  elapsed_s);
    //printf( "gflops/s : %f\n", gflops / elapsed_s < 0 ? -( gflops / elapsed_s ) : gflops / elapsed_s );

    err = cudaGetLastError();
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