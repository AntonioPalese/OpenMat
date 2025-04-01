#pragma once
#include "allocator.h"

namespace om
{
    // cpu allocator ////////////////////////////////// 
    template<typename T>
    T* CpuAllocator<T>::allocate(size_t count) {
        if (count == 0)
            return nullptr;

        T* ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
        if (!ptr)
            throw std::bad_alloc();
        return ptr;
    }

    template<typename T>
    void CpuAllocator<T>::deallocate(T* ptr) {
        std::free(ptr);
    }
    //////////////////////////////////

    // gpu allocator //////////////////////////////////
    template <typename T>
    inline T *GpuAllocator<T>::allocate(size_t count)
    {
        if (count == 0)
            return nullptr;

        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, sizeof(T) * count);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
        return ptr;
    }

    template <typename T>
    inline void GpuAllocator<T>::deallocate(T *ptr)
    {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }
    //////////////////////////////////
}
