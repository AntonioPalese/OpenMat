#include "matops.h"
#include "mat.h"
#include <math.h>
#include <stdio.h>
#include <chrono>


#define N pow(2.0, 14)
#define FLOPS  N*N*2.0*N
//#define FLOPS  2*N*N
#define GFLOPS (double)FLOPS*1.0e-9

template<typename Func>
void timer_ns(Func& f, Matrix A, Matrix B, Matrix C)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(f)(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double elapsed_s = elapsed_ns * 1e-9;
    printf( "elapsed : %f s\n",  elapsed_s);
    printf( "gflops/s : %f\n", GFLOPS / elapsed_s < 0 ? -( GFLOPS / elapsed_s ) : GFLOPS / elapsed_s );
}

void parallel_add()
{
    Matrix matA = make_matrix(N, N, 2.0f, true);    
    Matrix matB = make_matrix(N, N, 1.0f, true);    
    Matrix matC = make_matrix(N, N, true);   

    add(matA, matB, matC);
    
    //print(matC);

    delete_matrix(matA);
    delete_matrix(matB);
    delete_matrix(matC);
}

void sequencial_add()
{
    Matrix matA = make_matrix(N, N, 2.0f, false);    
    Matrix matB = make_matrix(N, N, 1.0f, false);    
    Matrix matC = make_matrix(N, N, false);   

    add(matA, matB, matC);
    
    print(matC);

    delete_matrix(matA);
    delete_matrix(matB);
    delete_matrix(matC);
}

void parallel_mul()
{
    Matrix matA = make_matrix(N, N, 2.0f, true);    
    Matrix matB = make_matrix(N, N, 1.0f, true);    
    Matrix matC = make_matrix(N, N, true);   

    mul(matA, matB, matC);
    
    //print(matC);

    delete_matrix(matA);
    delete_matrix(matB);
    delete_matrix(matC);
}

void sequencial_mul()
{
    Matrix matA = make_matrix(N, N, 2.0f, false);    
    Matrix matB = make_matrix(N, N, 1.0f, false);    
    Matrix matC = make_matrix(N, N, false);

    timer_ns(mul, matA, matB, matC);

    //print(matC);
    delete_matrix(matA);
    delete_matrix(matB);
    delete_matrix(matC);
}

int main(void)
{
    printf("################## device ##################\n");
    parallel_mul();
//     printf("##################  host  ##################\n");
//     sequencial_mul();
}