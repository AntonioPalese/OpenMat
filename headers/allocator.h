#pragma once

#include <cstdlib>
#include <stdexcept>

namespace om
{
    template<typename T>
    class Allocator
    {
    public:
        virtual T* allocate(size_t count) = 0;
        virtual void deallocate(T* ptr) = 0;
        virtual ~Allocator() = default;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const = 0;
    };

    // cpu allocator ////////////////////////////////// 
    template<typename T>
    class CpuAllocator : public Allocator<T>
    {
    public:
        virtual T* allocate(size_t count) override;
        virtual void deallocate(T* ptr) override;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const override;
    };

    template<typename T>
    class GpuAllocator : public Allocator<T>
    {
    public:
        virtual T* allocate(size_t count) override;
        virtual void deallocate(T* ptr) override;

        virtual void copyFromCurrentLoc(T* dst, const T* src, std::size_t count) const override;
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