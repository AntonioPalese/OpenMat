#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

float* make_matrix(size_t r, size_t c);

void delete_matrix(float *mat);

void set(float *mat, float val, size_t r, size_t c);

float* to_device(float *src, size_t r, size_t c);

void add(float *A, float *B, float *C, size_t r, size_t c);

void mul(float *A, float *B, float *C, size_t r, size_t c);

void sequencial_mul(float *A, float *B, float *C, size_t r, size_t c);

void cuda_print(float *mat, size_t r, size_t c);

void print(float *mat, size_t r, size_t c);

float* to_host(float *src, size_t r, size_t c);
