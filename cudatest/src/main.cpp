#include "matmul.h"


int main(void)
{
    constexpr const int N = 1024;
    float** matA = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
    float** matB = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);
    float** matC = set(make_matrix(N/2, N/2), 0.0f, N/2, N/2);

    float** gpu_matA = to_device(matA, N/2, N/2);
}