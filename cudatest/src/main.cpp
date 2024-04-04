#include "matops.h"
#include "mat.h"
#include <math.h>
#include <stdio.h>
#include <chrono>

#define N 32
#define FLOPS  N*N*2*N
#define GFLOPS (double)FLOPS*1.0e-9

void timer_ns( void( *func )( float*, float*, float*, size_t, size_t ), float* A, float* B, float* C, size_t r, size_t c )
{
    auto start = std::chrono::high_resolution_clock::now();
    ( *func )( A, B, C, r, c );
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double elapsed_s = elapsed_ns * 1e-9;
    printf( "elapsed : %f s\n",  elapsed_s);
    printf( "gflops/s : %f\n", GFLOPS / elapsed_s < 0 ? -( GFLOPS / elapsed_s ) : GFLOPS / elapsed_s );
}

void parallel_mul()
{
    int dim = N;

    Matrix matA = make_matrix(dim, dim, 2.0f, true);    
    Matrix matB = make_matrix(dim, dim, 1.0f, true);    
    Matrix matC = make_matrix(dim, dim, true);   

    add(matA, matB, matC);
    
    print(matC);

    delete_matrix(matA);
    delete_matrix(matB);
    delete_matrix(matC);



    // //add(matA_gpu, matB_gpu, matC_gpu, dim, dim);  
    // //timer_ns(&mul, matA_gpu, matB_gpu, matC_gpu, dim, dim);

    // timer_ns(&mul, matA_gpu, matB_gpu, matC_gpu, dim, dim);

    // //cuda_print(matC_gpu, dim, dim);
    
    // float* tmpC = to_host(matC_gpu, dim, dim);
    // //print(tmpC, dim, dim);

    // delete_matrix(matA_gpu);
    // delete_matrix(matB_gpu);
    // delete_matrix(matC_gpu);

    // free(matA);
    // free(matB);
    // free(matC);
}

// void sequencial_mul()
// {
//     size_t dim = (size_t)N;

//     float* matA = make_matrix(dim, dim);    
//     set(matA, 2.0f, dim, dim);

//     float* matB = make_matrix(dim, dim);    
//     set(matB, 1.0f, dim, dim);

//     float* matC = make_matrix(dim, dim);    
//     set(matC, 0.0f, dim, dim);   

//     timer_ns(&sequencial_mul, matA, matB, matC, dim, dim);

//     //print(matC, dim, dim);

//     free(matA);
//     free(matB);
//     free(matC);
// }

int main(void)
{
    parallel_mul();
    //sequencial_mul();
}