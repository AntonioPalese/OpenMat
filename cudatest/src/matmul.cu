#include "matmul.h"
#include "stdio.h"


/////// kernels //////////////

__global__ void _add(float** A, float **B, float **C, int r, int c)
{
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    C[y][x] = A[y][x]+B[y][x];
}
__global__ void _print(float** mat, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

////////////////////////////////

float** make_matrix(int r, int c)
{
    float** mat = (float**) malloc(r*sizeof(float*));
    for(int i = 0; i < r; i++)
    {
        mat[i] = (float*) malloc(c*sizeof(float));
    }
    return mat;
}

void delete_matrix(float** mat, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        free(mat[i]);
    }

    free(mat);
}

float** set(float**mat, float val, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            mat[i][j] = val;
        }
    }

    return mat;
}

__host__
void to_device(float **dst, float** src, int r, int c)
{
    cudaMalloc((void***)&dst, r*sizeof(float*));
    for(int i = 0; i < r; i++)
    {
        float* row;
        cudaMalloc((void**)&row, c*sizeof(float));
        cudaMemcpy(dst+i, row, sizeof(float*), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < r; i++)
    {
        cudaMemcpy(dst+i, src[i], c*sizeof(float), cudaMemcpyHostToDevice);        
    }
}

void add(float** A, float **B, float **C, int r, int c)
{    
    dim3 blkDim(1);
    dim3 thDim(c,r); // x, y, z
    _add<<<blkDim, thDim>>>(A, B, C, r, c);
}

void cuda_print(float** mat, int r, int c)
{
    _print<<<1, 1>>>(mat, r, c);
}

void print(float** mat, int r, int c)
{
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

void to_host(float **dst, float **src, int r, int c)
{
    dst = (float**) malloc(r*sizeof(float*));
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            dst[i] = (float*)malloc(c*sizeof(float));
        }
    }
}

// int main(void)
// {
//     constexpr const int N = 1024;
//     float** matA = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matB = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matC = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);

//     float** gpu_matA = to_device(matA, N/2, N/2);
// }