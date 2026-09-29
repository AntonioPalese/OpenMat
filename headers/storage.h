#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "mat_utils.h"
#include "allocator.h"
#include "stream.h"

namespace om
{
    // The buffer behind one or more Tensors.
    //
    // A Tensor is a (storage, offset, shape, stride) tuple; every view of a
    // tensor — reshape of a contiguous tensor, squeeze/unsqueeze, slice,
    // select — shares its Storage through a shared_ptr, and the last one to
    // die frees it.
    //
    // The stream-ownership invariant lives here, not in Tensor: memory from a
    // cudaMallocAsync pool must be freed on the stream it was allocated on, so
    // the Storage keeps that stream and its destructor frees on it, no matter
    // which view happens to be the last one alive. A view never frees on its
    // own stream.
    //
    // A Storage can also wrap memory OpenMat did not allocate (a tensor
    // imported through DLPack): it then holds a `release` callback instead of
    // freeing through the allocator, and the owner decides how and when the
    // memory goes. The allocator is still created, for the copy helpers.
    template <typename T>
    class Storage
    {
    public:
        Storage(size_t count, const Device& device, Stream stream, bool pinned)
            : m_Count(count), m_Device(device), m_Stream(std::move(stream)),
              m_Allocator(pinned ? AllocatorFactory<T>::create_pinned()
                                 : AllocatorFactory<T>::create(device.m_Dt))
        {
            m_Data = m_Allocator->allocate_async(count, m_Stream.get());
        }

        // Borrowed memory: `release` runs once, when the last view dies.
        Storage(T* data, size_t count, const Device& device, std::function<void()> release)
            : m_Data(data), m_Count(count), m_Device(device),
              m_Stream(Stream::default_stream()),
              m_Allocator(AllocatorFactory<T>::create(device.m_Dt)),
              m_Release(std::move(release))
        {}

        ~Storage()
        {
            if (m_Release)
                m_Release();
            else if (m_Data)
                m_Allocator->deallocate_async(m_Data, m_Stream.get());
        }

        Storage(const Storage&)            = delete;
        Storage& operator=(const Storage&) = delete;

        T* data() const { return m_Data; }
        size_t count() const { return m_Count; }
        const Device& device() const { return m_Device; }
        const Stream& stream() const { return m_Stream; }
        Allocator<T>& allocator() const { return *m_Allocator; }

    private:
        T* m_Data = nullptr;
        size_t m_Count;
        Device m_Device;
        Stream m_Stream;
        std::unique_ptr<Allocator<T>> m_Allocator;
        std::function<void()> m_Release;
    };
}
