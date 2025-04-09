
#include "tensor.cuh"
#include <numeric>

template<typename value_type>
om::Tensor<value_type>::Tensor(const std::vector<size_t>& shape, const Device & dv): m_Shape(shape), m_Device(dv), m_Allocator(AllocatorFactory<value_type>::create(dv.m_Dt))
{
    _compute_strides();
    size_t total_size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    m_Data = m_Allocator->allocate(total_size_);
}

template<typename value_type>
om::Tensor<value_type>::Tensor(const Tensor& rhs) : m_Shape(rhs.m_Shape), m_Stride(rhs.m_Stride), m_Device(rhs.m_Device), m_Allocator(AllocatorFactory<value_type>::create(rhs.m_Device.m_Dt))
{
    size_t total_size_ = std::accumulate(m_Shape.begin(), m_Shape.end(), 1, std::multiplies<>());
    m_Data = m_Allocator->allocate(total_size_);
    m_Allocator->copy(m_Data, rhs.m_Data, total_size_);
}

template <typename value_type>
om::Tensor<value_type>::Tensor(Tensor &&rhs) : m_Shape(rhs.m_Shape), m_Stride(rhs.m_Stride), m_Device(rhs.m_Device)
{
    if(rhs.m_Data)
    {
        m_Allocator = std::move(rhs.m_Allocator);
        size_t total_size_ = std::accumulate(m_Shape.begin(), m_Shape.end(), 1, std::multiplies<>());
        m_Data = m_Allocator->allocate(total_size_);
        m_Allocator->copy(m_Data, rhs.m_Data, total_size_);

        rhs.m_Allocator->deallocate(rhs.m_Data);
        rhs.m_Data = nullptr;
    }
}

template <typename value_type>
om::Tensor<value_type>::~Tensor()
{
    if(m_Data)
        m_Allocator->deallocate(m_Data);
}

template <typename value_type>
void om::Tensor<value_type>::copyToHost(value_type *dest) const
{            
    size_t total_size_ = std::accumulate(m_Shape.begin(), m_Shape.end(), 1, std::multiplies<>());
    if(device_type() == DEVICE_TYPE::CUDA)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, total_size_);
    else
        std::memcpy(dest, m_Data, sizeof(value_type) * total_size_); 
}

template <typename value_type>
void om::Tensor<value_type>::copyToDevice(value_type *dest) const
{
    size_t total_size_ = std::accumulate(m_Shape.begin(), m_Shape.end(), 1, std::multiplies<>());
    if(device_type() == DEVICE_TYPE::CPU)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, total_size_);
    else
        throw std::runtime_error("Tensor::copyToDevice: memory already on device");
}

template <typename _Ty>
inline void om::Tensor<_Ty>::_compute_strides()
{
    m_Stride.resize(m_Shape.size());
    m_Stride.back() = 1;
    for (int i = m_Shape.size() - 2; i >= 0; --i) {
        m_Stride[i] = m_Stride[i + 1] * m_Shape[i + 1];
    }
}

template <typename _Ty>
inline size_t om::Tensor<_Ty>::_compute_flat_index(const std::vector<size_t> &indices) const
{
    if (indices.size() != m_Shape.size()) 
        throw std::out_of_range("Tensor access: number of indices does not match tensor rank.");
    
    size_t flat_index = 0;
    size_t i = 0;
    for (size_t idx : indices) 
    {
        if (idx >= m_Shape[i])
            throw std::out_of_range("Tensor access: index out of bounds at dimension " + std::to_string(i));

        flat_index += idx * m_Stride[i];
        ++i;
    }
    return flat_index;
}
