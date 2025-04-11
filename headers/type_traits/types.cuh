#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct float16_t {
    __half value;
    
    __host__ __device__
    float16_t() : value(__float2half(0.0f)) {}
    
    __host__ __device__
    float16_t(float f) : value(__float2half(f)) {}
    
    __host__ __device__
    float16_t(const __half& h) : value(h) {}
    
    __host__ __device__
    operator float() const { return __half2float(value); }
    
    __host__ __device__
    operator __half() const { return value; }
};

template <typename T>
__device__ inline T device_load(const T* ptr) {
    return __ldg(ptr);
}

template <>
__device__ inline float16_t device_load(const float16_t* ptr) {
#if __CUDA_ARCH__ >= 300
    const __half* raw = reinterpret_cast<const __half*>(ptr);
    return float16_t(__ldg(raw));
#else
    return *ptr;
#endif
}

__host__ __device__ inline
float16_t operator+(const float16_t &lhs, const float16_t &rhs)
{
    #if __CUDA_ARCH__ >= 530
            return float16_t(__hadd(lhs.value, rhs.value));
    #else
            return float(lhs) + float(rhs);
    #endif
}

__host__ __device__ inline
float16_t operator-(const float16_t &lhs, const float16_t &rhs)
{
    #if __CUDA_ARCH__ >= 530
            return float16_t(__hsub(lhs.value, rhs.value));
    #else
            return float(lhs) - float(rhs);
    #endif
}

__host__ __device__ inline
float16_t operator*(const float16_t &lhs, const float16_t &rhs)
{
    #if __CUDA_ARCH__ >= 530
            return float16_t(__hmul(lhs.value, rhs.value));
    #else
            return float(lhs) * float(rhs);
    #endif
}

__host__ __device__ inline
float16_t operator/(const float16_t &lhs, const float16_t &rhs)
{
    #if __CUDA_ARCH__ >= 530
            return float16_t(__hdiv(lhs.value, rhs.value));
    #else
            return float(lhs) / float(rhs);
    #endif
}
    
template <>
struct std::is_arithmetic<float16_t> : std::true_type {};