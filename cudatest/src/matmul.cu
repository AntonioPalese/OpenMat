#include "matmul.h"
#include "stdio.h"
#include "cuda.h"

///////////// Kernels /////////////////////////

__global__ void _add(float* A, float *B, float *C, size_t r, size_t c)
{
    //printf("kernel _add\n");
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    C[x+c*y] = A[x+c*y] + B[x+c*y];
}

__global__ void _mul(float* A, float *B, float *C, size_t r, size_t c)
{
    //printf("kernel _mul\n");
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

   
    float value = 0;
    for(int i = 0; i < c; i++)
    {        
        value += A[i+c*y] * B[x+c*i];
    }
    C[x+c*y] = value;    
}

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

///////////////////////////////////////////////

void print_cuda_err(cudaError_t err, const char* tag)
{
    if(err != cudaSuccess)
        printf("[CUDA ERROR (%s)] :  %s\n", tag, cudaGetErrorString(err));
}

float* make_matrix(size_t r, size_t c)
{
    float* mat = (float*) malloc(r*c*sizeof(float));
    return mat;
}

void delete_matrix(float* mat)
{
    cudaError_t err = cudaFree(mat);
    print_cuda_err(err, "cudaFree");
}

void set(float*mat, float val, size_t r, size_t c)
{
    for(int i = 0; i < r*c; i++)
    {
        mat[i] = val;
    }    
}

float* to_device(float* src, size_t r, size_t c)
{
    float* dst;
    cudaDeviceSynchronize();
    cudaError_t err =  cudaMalloc((void**)&dst, r*c*sizeof(float));
    print_cuda_err(err, "cudaMalloc");
    err = cudaMemcpy(dst, src, r*c*sizeof(float), cudaMemcpyHostToDevice);
    print_cuda_err(err, "cudaMemcpy");
    return dst;
}

void add(float* A, float *B, float *C, size_t r, size_t c)
{    
    dim3 thDim(c,r); // x, y, z
    _add<<<1, thDim>>>(A, B, C, r, c);
    cudaError_t err = cudaGetLastError();
    print_cuda_err(err, "_add kernel");
    err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");
    
}

void mul(float* A, float *B, float *C, size_t r, size_t c)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(c / dimBlock.x, r / dimBlock.y); // x, y, z
    _mul<<<dimGrid, dimBlock>>>(A, B, C, r, c);
    cudaError_t err = cudaGetLastError();
    print_cuda_err(err, "_mul kernel");
    err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");
}

void sequencial_mul(float* A, float *B, float *C, size_t r, size_t c)
{    
    for(int i = 0; i < r; i++)
    {        
        for(int j = 0; j < c; j++)
        {
            float acc = 0;
            for(int k = 0; k < r; k++)
            {
                acc += A[k+i*c] * B[j+k*c];
            }
            C[j+i*c] = acc;
        }
    }    
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

float* to_host(float *src, size_t r, size_t c)
{
    float* dst = (float*) malloc(r*c*sizeof(float));
    cudaError_t err = cudaMemcpy(dst, src, r*c*sizeof(float), cudaMemcpyDeviceToHost);
    print_cuda_err(err, "cudaMemcpy");
    err = cudaDeviceSynchronize();
    print_cuda_err(err, "cudaDeviceSynchronize");
    return dst;
}

