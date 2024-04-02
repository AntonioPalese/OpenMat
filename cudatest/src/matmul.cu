#include "matmul.h"

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
float ** to_device(float** mat, int r, int c)
{
    float** gpu_mat;
    cudaMalloc((void***)&gpu_mat, r*sizeof(float*));
    for(int i = 0; i < r; i++)
    {
        float* row;
        cudaMalloc((void**)&row, c*sizeof(float));
        cudaMemcpy(gpu_mat+i, row, sizeof(float*), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < r; i++)
    {
        cudaMemcpy(gpu_mat+i, mat[i], c*sizeof(float), cudaMemcpyHostToDevice);        
    }

    return gpu_mat;
}

// int main(void)
// {
//     constexpr const int N = 1024;
//     float** matA = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matB = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
//     float** matC = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);

//     float** gpu_matA = to_device(matA, N/2, N/2);
// }