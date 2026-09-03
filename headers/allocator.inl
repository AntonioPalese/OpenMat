#include "allocator.h"
#include "cuda_defines.cuh"
#include "host_pool.h"

namespace om
{
    // Host memory goes through om::detail::HostPool rather than malloc/free:
    // above glibc's 128 KB mmap threshold a plain malloc costs a page fault
    // per 4 KB of the buffer on first touch, which is the whole of the CPU
    // elementwise and D2H gap against NumPy/PyTorch. See host_pool.h.
    template<typename T>
    T* CpuAllocator<T>::allocate(size_t count) {
        if (count == 0)
            return nullptr;
        if (count > static_cast<size_t>(-1) / sizeof(T))
            throw std::bad_alloc();

        return static_cast<T*>(detail::HostPool::instance().allocate(sizeof(T) * count));
    }

    template<typename T>
    void CpuAllocator<T>::deallocate(T* ptr) {
        detail::HostPool::instance().deallocate(ptr);
    }

    template <typename T>
    void CpuAllocator<T>::copy(T *dst, const T *src, size_t count)
    {
        void* res = std::memcpy((void*)dst, (const void*)src, count*sizeof(T));
        if(!res)
            throw std::bad_alloc();
    }
    
    template <typename T>
    void CpuAllocator<T>::copyFromCurrentLoc(T *dst, const T *src, std::size_t count) const
    {
        CUDA_CALL(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    template <typename T>
    void CpuAllocator<T>::copy_host_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream)
    {
        // cudaMemcpyAsync with a non-pinned src will silently fall back to synchronous
        // copy on many drivers, but is safe to call unconditionally.
        CUDA_CALL(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
    }
    

    
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

    template <typename T>
    T* GpuAllocator<T>::allocate_async(size_t count, cudaStream_t stream)
    {
        if (count == 0) return nullptr;
        T* ptr = nullptr;
#if CUDART_VERSION >= 11020
        CUDA_CALL(cudaMallocAsync((void**)&ptr, sizeof(T) * count, stream));
#else
        CUDA_CALL(cudaMalloc((void**)&ptr, sizeof(T) * count));
#endif
        return ptr;
    }

    template <typename T>
    void GpuAllocator<T>::deallocate_async(T* ptr, cudaStream_t stream)
    {
#if CUDART_VERSION >= 11020
        CUDA_CALL(cudaFreeAsync(ptr, stream));
#else
        CUDA_CALL(cudaFree(ptr));
#endif
    }

    template <typename T>
    void GpuAllocator<T>::copy_async(T* dst, const T* src, size_t count, cudaStream_t stream)
    {
        CUDA_CALL(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream));
    }

    template <typename T>
    void GpuAllocator<T>::copy_host_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream)
    {
        CUDA_CALL(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
    }

    template <typename T>
    void GpuAllocator<T>::copy_device_to_host_async(T* dst, const T* src, size_t count, cudaStream_t stream)
    {
        CUDA_CALL(cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
    }
}
