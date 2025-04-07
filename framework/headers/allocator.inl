#pragma once
#include "allocator.h"
#include "cuda_defines.cuh"

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

    template <typename T>
    void CpuAllocator<T>::copy(T *dst, const T *src, size_t count)
    {
        T* res = std::memcpy(dst, src, count*sizeof(T));
        if(!res)
            throw std::bad_alloc();
    }

    template <typename T>
    void CpuAllocator<T>::copyFromCurrentLoc(T *dst, const T *src, std::size_t count) const
    {
        CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    //////////////////////////////////

    // gpu allocator //////////////////////////////////
    template <typename T>
    T *GpuAllocator<T>::allocate(size_t count)
    {
        if (count == 0)
            return nullptr;

        T* ptr = nullptr;
        CUDA_CALL(cudaMalloc((void**)&ptr, sizeof(T) * count));
        return ptr;
    }

    template <typename T>
    void GpuAllocator<T>::deallocate(T *ptr)
    {
        CUDA_CALL(cudaFree(ptr));
    }
    
    template <typename T>
    void GpuAllocator<T>::copy(T *dst, const T *src, size_t count)
    {
       CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
    }
    
    template <typename T>
    void GpuAllocator<T>::copyFromCurrentLoc(T *dst, const T *src, std::size_t count) const
    {
       CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }
    //////////////////////////////////
}
