#include <cuda.h>
#include <stdio.h>
#define BLOCK_SIZE 4

struct Matrix
{
    int rows;
    int cols;
    int stride;
    float* data;
};


__host__ __device__
float const_at(const Matrix* m, int r, int c)
{
    return m->data[c+r*m->stride];
}

__host__ __device__
float* at(Matrix* m, int r, int c)
{
    return m->data + c+r*m->stride;
}


__device__ __host__
void print_cuda_err(cudaError_t err, const char* tag = "")
{
    if(err != cudaSuccess)
        printf("[CUDA ERROR (%s)] :  %s\n", tag, cudaGetErrorString(err));
}

__device__
Matrix subMat(Matrix src, int r, int c)
{
    Matrix dst{BLOCK_SIZE, BLOCK_SIZE, src.stride, src.data + (c*BLOCK_SIZE+r*BLOCK_SIZE*src.stride)};
    return dst;
}

__global__ void kernel(Matrix M)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix subM = subMat(M, blockRow, blockCol);

    __shared__ float sM[BLOCK_SIZE][BLOCK_SIZE];
    sM[threadIdx.y][threadIdx.x] = blockIdx.x + blockIdx.y * gridDim.x;
    //printf("(%d, %d)\n", blockIdx.x, blockIdx.y);
    //printf("(%d)\n", blockIdx.x + blockIdx.y * gridDim.x);

    __syncthreads();

    *at(&subM, threadIdx.y, threadIdx.x) = sM[threadIdx.y][threadIdx.x];
}

void print(const Matrix m)
{
    for(int i = 0; i < m.rows; i++)
    {
        for(int j = 0; j < m.cols; j++)
        {
            printf("%f ", const_at(&m, i, j));
        }
        printf("\n");
    }
}

int main(void)
{
    int N = 8;
    cudaError_t err;

    float* data;
    size_t size = N*N*sizeof(float);
    err = cudaMalloc(&data, size); 
    print_cuda_err(err);
    Matrix M{N, N, N, data};

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(M.cols / BLOCK_SIZE, M.rows / BLOCK_SIZE);

    kernel<<<gridDim, blockDim>>>(M); 
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    print_cuda_err(err);

    
    Matrix res{N, N, N, (float*)malloc(size)};
    err = cudaMemcpy(res.data, M.data, size, cudaMemcpyDeviceToHost);
    print_cuda_err(err);
    print(res);

}

