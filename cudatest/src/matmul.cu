#include "matmul.h"
#include "stdio.h"


///////////// Kernels /////////////////////////

__global__ void _add(float* A, float *B, float *C, int r, int c)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    C[x+c*y] = A[x+c*y] + B[x+c*y];
}

__global__ void _print(float* A, int r, int c)
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

float* make_matrix(int r, int c)
{
    float* mat = (float*) malloc(r*c*sizeof(float));
    return mat;
}

void delete_matrix(float* mat)
{
    free(mat);
}

void set(float*mat, float val, int r, int c)
{
    for(int i = 0; i < r*c; i++)
    {
        mat[i] = val;
    }    
}

float* to_device(float* src, int r, int c)
{
    float* dst;
    cudaMalloc((void**)&dst, r*c*sizeof(float));
    cudaMemcpy(dst, src, r*c*sizeof(float), cudaMemcpyHostToDevice);
    return dst;
}

void add(float* A, float *B, float *C, int r, int c)
{    
    dim3 blkDim(1);
    dim3 thDim(c,r); // x, y, z
    _add<<<blkDim, thDim>>>(A, B, C, r, c);
}

void cuda_print(float* mat, int r, int c)
{
    _print<<<1, 1>>>(mat, r, c);
}

void print(float* mat, int r, int c)
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

float* to_host(float *src, int r, int c)
{
    float* dst = (float*) malloc(r*c*sizeof(float));
    cudaMemcpy(dst, src, r*c*sizeof(float), cudaMemcpyDeviceToHost);
    return dst;
}

// int main(void)
// {
//     constexpr const int N = 1024;
//     float** matA = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matB = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matC = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);

//     float** gpu_matA = to_device(matA, N/2, N/2);
// }