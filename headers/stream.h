#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace om
{

class Stream
{
    cudaStream_t m_stream = nullptr;
    bool         m_owns   = false;

public:
    // Creates a new CUDA stream (owns it)
    Stream()
    {
        cudaError_t err = cudaStreamCreate(&m_stream);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("[om::Stream] cudaStreamCreate failed: ") +
                cudaGetErrorString(err));
        m_owns = true;
    }

    // Wraps an existing cudaStream_t without taking ownership
    explicit Stream(cudaStream_t s) : m_stream(s), m_owns(false) {}

    ~Stream()
    {
        if (m_owns && m_stream)
            cudaStreamDestroy(m_stream);
    }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept
        : m_stream(other.m_stream), m_owns(other.m_owns)
    {
        other.m_stream = nullptr;
        other.m_owns   = false;
    }

    Stream& operator=(Stream&& other) noexcept
    {
        if (this != &other) {
            if (m_owns && m_stream) cudaStreamDestroy(m_stream);
            m_stream       = other.m_stream;
            m_owns         = other.m_owns;
            other.m_stream = nullptr;
            other.m_owns   = false;
        }
        return *this;
    }

    void synchronize() const
    {
        cudaError_t err = cudaStreamSynchronize(m_stream);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("[om::Stream] cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
    }

    cudaStream_t get() const { return m_stream; }

    // Returns a non-owning wrapper around the default (null) stream
    static Stream default_stream() { return Stream(cudaStream_t{nullptr}); }
};

} // namespace om
