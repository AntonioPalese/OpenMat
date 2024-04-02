#include "matmul.h"


int main(void)
{
    constexpr const int N = 1024;
    float** matA = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
    float** matB = set(make_matrix(N/2, N/2), 1.0f, N/2, N/2);
    float** matC = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);


    print(matB, N/2, N/2);
    //print(matC, N/2, N/2);

    float** gpu_matA;
    to_device(gpu_matA, matA, N/2, N/2);
    float** gpu_matB;
    to_device(gpu_matB, matB, N/2, N/2);
    float** gpu_matC;
    to_device(gpu_matC, matC, N/2, N/2);  

    float** tmpA;
    to_host(tmpA, gpu_matA, N/2, N/2);

    add(gpu_matA, gpu_matB, gpu_matC, N/2, N/2);

    //print(gpu_matA, N/2, N/2);
}