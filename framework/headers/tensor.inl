
#include "tensor.cuh"

template<typename value_type>
om::Tensor<value_type>::Tensor(const std::vector<size_t>& shape, const Device & dv): m_Shape(shape), m_Device(dv), m_Allocator(AllocatorFactory<value_type>::create(dv.m_Dt))
{
    _compute_strides();
    size_t total_size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    m_Allocator->allocate(total_size_);
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