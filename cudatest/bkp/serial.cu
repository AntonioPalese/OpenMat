#include <stdio.h>
#include <chrono>
#include <iostream>

void m_set(float* src, float val, int size)
{
    for(int i = 0; i < size; i++)
    {
        src[i] = val;
    }
}

void add(float*A, float*B, float*C, int N)
{
    for(int i = 0; i < N; i++)
    {
        C[i] = A[i]+B[i];
    }
}

void serial(int N)
{
    float* host_A = (float*)malloc(N*sizeof(float));
    float* host_B = (float*)malloc(N*sizeof(float));
    float* host_C = (float*)malloc(N*sizeof(float));

    m_set(host_A, 0.0f, N);
    m_set(host_B, 11.0f, N);
    m_set(host_C, 0.0f, N);

    auto start = std::chrono::high_resolution_clock::now();
    add(host_A, host_B, host_C, N);
    auto end = std::chrono::high_resolution_clock::now();


    std::cout << "function elapsed time : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " micosecs" << std::endl;

    free(host_A);
    free(host_B);
    free(host_C);    
    
}

int main()
{    
    const int N = 1024*10000;
    serial(N);    
}