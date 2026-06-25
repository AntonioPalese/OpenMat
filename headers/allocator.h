#pragma once

#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

namespace om
{
    template<typename T>
    class Allocator
    {
    public:
        virtual ~Allocator() = default;

        virtual T* allocate(size_t count) = 0;
        virtual void deallocate(T* ptr) = 0;

        virtual void copy(T* dst, const T* src, size_t count) = 0;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const = 0;

        // Stream-aware async variants. Default implementations fall back to the
        // synchronous methods so subclasses only override what they support.
        virtual T* allocate_async(size_t count, cudaStream_t stream) {
            return this->allocate(count);
        }
        virtual void deallocate_async(T* ptr, cudaStream_t stream) {
            this->deallocate(ptr);
        }
        virtual void copy_async(T* dst, const T* src, size_t count, cudaStream_t stream) {
            this->copy(dst, src, count);
        }
        virtual void copy_host_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream) {
            // Default: synchronous H2D copy (overridden by CpuAllocator for pinned src)
            this->copyFromCurrentLoc(dst, src, count);
        }
        virtual void copy_device_to_host_async(T* dst, const T* src, size_t count, cudaStream_t stream) {
            // Default: synchronous D2H copy (overridden by GpuAllocator)
            this->copyFromCurrentLoc(dst, src, count);
        }
    };

    template<typename T>
    class CpuAllocator : public Allocator<T>
    {
    public:
        virtual T* allocate(size_t count) override;
        virtual void deallocate(T* ptr) override;

        virtual void copy(T* dst, const T* src, size_t count) override;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const override;

        virtual void copy_host_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream) override;
    };

    template<typename T>
    class GpuAllocator : public Allocator<T>
    {
    public:
        virtual T* allocate(size_t count) override;
        virtual void deallocate(T* ptr) override;

        virtual void copy(T* dst, const T* src, size_t count) override;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const override;

        virtual T* allocate_async(size_t count, cudaStream_t stream) override;
        virtual void deallocate_async(T* ptr, cudaStream_t stream) override;
        virtual void copy_async(T* dst, const T* src, size_t count, cudaStream_t stream) override;
        virtual void copy_host_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream) override;
        virtual void copy_device_to_host_async(T* dst, const T* src, size_t count, cudaStream_t stream) override;
    };

    template<typename T>
    class AllocatorFactory
    {
    public:
        static std::unique_ptr<Allocator<T>> create(DEVICE_TYPE devType)
        {
            switch (devType)
            {
                case DEVICE_TYPE::CPU : return std::make_unique<CpuAllocator<T>>();           
                case DEVICE_TYPE::CUDA : return std::make_unique<GpuAllocator<T>>();           
            default:
                throw std::invalid_argument("Unknown AllocatorType");
            }
        }
    };
}

#include "allocator.inl"