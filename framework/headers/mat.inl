#pragma once
#include "mat.cuh"
#include "allocator.h"


template <typename _Ty>
om::Mat<_Ty>::Mat(int r, int c, const Device& dt): m_Rows(r), m_Cols(c), m_Device(dt), m_Allocator(AllocatorFactory<_Ty>::create(dt.m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
}

template <typename _Ty>
om::Mat<_Ty>::Mat(const Mat& rhs) : m_Rows(rhs.m_Rows), m_Cols(rhs.m_Cols), m_Allocator(AllocatorFactory<_Ty>::create(rhs.device().m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
    // ?
    std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols);
}

template <typename _Ty>
om::Mat<_Ty>::Mat(Mat &&rhs)
{
    if(rhs.m_Data)
    {
        m_Allocator = std::move(rhs.m_Allocator);
        m_Data = m_Allocator->allocate(m_Rows*m_Cols);
        // ?
        std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols);

        rhs.m_Allocator->deallocate(rhs.m_Data);
        rhs.m_Data = nullptr;
    }
}

template <typename _Ty>
om::Mat<_Ty>::~Mat()
{
    if(m_Data)
        m_Allocator->deallocate(m_Data);
}

template <typename _Ty>
void om::Mat<_Ty>::copyToHost(value_type *dest) const
{            
    if(device_type() == DEVICE_TYPE::CUDA)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, m_Rows*m_Cols);
    else
        std::memcpy(dest, m_Data, sizeof(value_type) * m_Rows * m_Cols); 
}

template <typename _Ty>
void om::Mat<_Ty>::copyToDevice(value_type *dest) const
{
    if(device_type() == DEVICE_TYPE::CPU)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, m_Rows*m_Cols);
    else
        throw std::runtime_error("Mat::copyToDevice: memory already on device");
}