#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK                                         \
    do {                                                   \
        cudaError_t err = cudaGetLastError();              \
        if (err != cudaSuccess)                            \
            throw std::runtime_error(                      \
                std::string("[CUDA ERROR] : ") +           \
                cudaGetErrorString(err));                  \
    } while (0)                                           


#define CUDA_CALL(expr)                                    \
    do {                                                   \
        cudaError_t err = (expr);                          \
        if (err != cudaSuccess)                            \
            throw std::runtime_error(                      \
                std::string("[CUDA ERROR] : ") +           \
                cudaGetErrorString(err));                  \
    } while (0)
    

inline bool is_device_pointer(const void* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

#if CUDART_VERSION >= 10000
    // Since CUDA 10, must check return status and memoryType
    if (err != cudaSuccess) return false;
    return attr.type == cudaMemoryTypeDevice;
#else
    // For CUDA < 10
    if (err != cudaSuccess) return false;
    return attr.memoryType == cudaMemoryTypeDevice;
#endif
}
