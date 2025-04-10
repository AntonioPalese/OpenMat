#include "mat.h"
#include <string.h>
#include <stdio.h>

Matrix make_matrix(int r, int c, bool cuda)
{
    float* elements = (float*) malloc(r*c*sizeof(float));
    memset(elements, 0, r*c*sizeof(float));
    Matrix M { r, c, c, elements, cuda };
    return M;
}

Matrix make_matrix(int r, int c, float init_val, bool cuda)
{
    Matrix M = make_matrix(r, c, cuda);
    m_set(M, init_val);
    return M;
}

void delete_matrix(Matrix m)
{
    m.cols = 0;
    m.rows = 0;
    m.stride = 0;
    free(m.data);
    m.data = NULL;
    m.cuda = false;    
}


void to_device(Matrix m)
{
    m.cuda = true;
}

void to_host(Matrix m)
{
    m.cuda = false;
}

void print(const Matrix m)
{
    for(int i = 0; i < m.rows; i++)
    {
        for(int j = 0; j < m.cols; j++)
        {
            printf("%f ", *at(m, i, j));
        }
        printf("\n");
    }
}

float* at(Matrix M, int r, int c)
{
    return M.data + c+ r * M.cols;
}

void m_set(Matrix M, float val)
{
    for(int i = 0; i < M.rows; i++)
    {
        for(int j = 0; j < M.cols; j++)
        {
            *at(M, i, j) = val;
        }
    }
}


