#pragma once
#include <stdlib.h>
#include <stdint.h>


struct Matrix
{
    int rows;
    int cols;
    int stride;
    float* data;
    bool cuda = false;
};

Matrix make_matrix(int r, int c, float init_val, bool cuda = false);
Matrix make_matrix( int r, int c, bool cuda = false);

void delete_matrix(Matrix m);

void to_device(Matrix m);
void to_host(Matrix m);

void print(const Matrix m);

float *at(Matrix M, int r, int c);

void m_set(Matrix M, float val);




