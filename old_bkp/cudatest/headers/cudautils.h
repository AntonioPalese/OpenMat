#pragma once
#include "mat.h"

void print_cuda_err(cudaError_t err, const char* tag);

void to_device(Matrix* dst, const Matrix* src);

void to_host(Matrix *dst, const Matrix *src);

void cuda_print(float* mat, size_t r, size_t c);

void print(float* mat, size_t r, size_t c);