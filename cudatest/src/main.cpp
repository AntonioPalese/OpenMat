#include "matmul.h"


int main(void)
{
    constexpr const int N = 16;

    float* matA = make_matrix(N/2, N/2);    
    set(matA, 0.0f, N/2, N/2);

    float* matB = make_matrix(N/2, N/2);    
    set(matB, 1.0f, N/2, N/2);

    float* matC = make_matrix(N/2, N/2);    
    set(matC, 0.0f, N/2, N/2);   

    //print(matB, N/2, N/2); 
    float* matA_gpu = to_device(matA, N/2, N/2);
    float* matB_gpu = to_device(matB, N/2, N/2);
    float* matC_gpu = to_device(matC, N/2, N/2); 

    cuda_print(matA_gpu, N/2, N/2);
    add(matA_gpu, matB_gpu, matC_gpu, N/2, N/2);   
    
    float* tmpC = to_host(matC_gpu, N/2, N/2);
    print(tmpC, N/2, N/2);
}