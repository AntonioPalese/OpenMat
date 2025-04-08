#pragma once
#include "mat.cuh"
#include "allocator.h"


template <typename value_type>
om::Mat<value_type>::Mat(int r, int c, const Device& dt): m_Rows(r), m_Cols(c), m_Device(dt), m_Allocator(AllocatorFactory<value_type>::create(dt.m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
}

template <typename value_type>
om::Mat<value_type>::Mat(const Mat& rhs) : m_Rows(rhs.m_Rows), m_Cols(rhs.m_Cols), m_Device(rhs.m_Device), m_Allocator(AllocatorFactory<value_type>::create(rhs.device().m_Dt))
{
    m_Data = m_Allocator->allocate(m_Rows*m_Cols);
    // ?
    std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols*sizeof(value_type));
}

template <typename value_type>
om::Mat<value_type>::Mat(Mat &&rhs) : m_Rows(rhs.m_Rows), m_Cols(rhs.m_Cols), m_Device(rhs.device())
{
    if(rhs.m_Data)
    {
        m_Allocator = std::move(rhs.m_Allocator);
        m_Data = m_Allocator->allocate(m_Rows*m_Cols);
        // ?
        std::memcpy(m_Data, rhs.m_Data, m_Rows*m_Cols*sizeof(value_type));

        rhs.m_Allocator->deallocate(rhs.m_Data);
        rhs.m_Data = nullptr;
    }
}

template <typename value_type>
om::Mat<value_type>::~Mat()
{
    if(m_Data)
        m_Allocator->deallocate(m_Data);
}

template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::add(const Mat<value_type> &rhs) const
{
    if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) {
        throw std::runtime_error("add: dimension mismatch");
    }
    if (this->device_type() != rhs.device_type()) {
        throw std::runtime_error("add: device mismatch");
    }

    Mat<value_type> out(this->rows(), this->cols(), this->device());

    _add(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}
template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::operator+(const Mat<value_type> &rhs) const
{
    return this->add(rhs);
}

template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::sub(const Mat<value_type> &rhs) const
{
    if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) {
        throw std::runtime_error("sub: dimension mismatch");
    }
    if (this->device_type() != rhs.device_type()) {
        throw std::runtime_error("sub: device mismatch");
    }

    Mat<value_type> out(this->rows(), this->cols(), this->device());

    _sub(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}
template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::operator-(const Mat<value_type> &rhs) const
{
    return this->sub(rhs);
}

template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::mul(const Mat<value_type> &rhs) const
{
    if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) {
        throw std::runtime_error("sub: dimension mismatch");
    }
    if (this->device_type() != rhs.device_type()) {
        throw std::runtime_error("sub: device mismatch");
    }

    Mat<value_type> out(this->rows(), this->cols(), this->device());

    _mul(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}
template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::operator*(const Mat<value_type> &rhs) const
{
    return this->mul(rhs);
}

template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::div(const Mat<value_type> &rhs) const
{
    if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) {
        throw std::runtime_error("sub: dimension mismatch");
    }
    if (this->device_type() != rhs.device_type()) {
        throw std::runtime_error("sub: device mismatch");
    }

    Mat<value_type> out(this->rows(), this->cols(), this->device());

    _div(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}
template <typename value_type>
om::Mat<value_type> om::Mat<value_type>::operator/(const Mat<value_type> &rhs) const
{
    return this->div(rhs);
}

template <typename value_type>
void om::Mat<value_type>::copyToHost(value_type *dest) const
{            
    if(device_type() == DEVICE_TYPE::CUDA)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, m_Rows*m_Cols);
    else
        std::memcpy(dest, m_Data, sizeof(value_type) * m_Rows * m_Cols); 
}

template <typename value_type>
void om::Mat<value_type>::copyToDevice(value_type *dest) const
{
    if(device_type() == DEVICE_TYPE::CPU)
        m_Allocator->copyFromCurrentLoc(dest, m_Data, m_Rows*m_Cols);
    else
        throw std::runtime_error("Mat::copyToDevice: memory already on device");
}