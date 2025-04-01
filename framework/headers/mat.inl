#pragma once
#include "mat.cuh"
#include "allocator.h"


template <typename _Ty>
om::Mat<_Ty>::Mat(int r, int c, const Device& dt): m_Rows(r), m_Cols(c), m_Allocator(AllocatorFactory<_Ty>::create(dt.m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
}

template <typename _Ty>
inline om::Mat<_Ty>::Mat(const Mat& rhs) : m_Rows(rhs.m_Rows), m_Cols(rhs.m_Cols), m_Allocator(AllocatorFactory<_Ty>::create(rhs.device().m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
    std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols);
}

template <typename _Ty>
inline om::Mat<_Ty>::Mat(Mat &&rhs)
{
    if(rhs.m_Data)
    {
        m_Allocator = std::move(rhs.m_Allocator);
        m_Data = m_Allocator->allocate(m_Rows*m_Cols);
        std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols);

        rhs.m_Allocator->deallocate(rhs.m_Data);
        rhs.m_Data = nullptr;
    }
}

template <typename _Ty>
inline om::Mat<_Ty>::~Mat()
{
    if(m_Data)
        m_Allocator->deallocate(m_Data);
}
