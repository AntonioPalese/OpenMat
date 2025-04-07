#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <memory>
#include <vector>

#include "mat_utils.h"
#include "allocator.h"
#include "mat_view.cuh"
#include "kernel_launcher.h"


namespace om
{
    template<typename _Ty>
    class Tensor
    {
    public:
        using value_type = _Ty;

        Tensor(const std::vector<size_t>& shape, const Device& dv = Device(0, DEVICE_TYPE::CPU));
        Tensor(const Tensor& rhs); // copy
        Tensor(Tensor&& rhs); // move
        ~Tensor();        

        /*
        inline const value_type& operator()(int r, int c) const
        {
            if (r < 0 || r >= m_Rows || c < 0 || c >= m_Cols)
                throw std::out_of_range("Matrix index out of bounds");
            return m_Data[r * m_Cols + c];
        }
        inline value_type& operator()(int r, int c)
        {
            if (r < 0 || r >= m_Rows || c < 0 || c >= m_Cols)
                throw std::out_of_range("Matrix index out of bounds");
            return m_Data[r * m_Cols + c];
        }

        __host__ MatView<value_type> view() {
            return MatView<value_type>{m_Data, m_Rows, m_Cols};
        }
    
        __host__ MatView<const value_type> view() const {
            return MatView<const value_type>{m_Data, m_Rows, m_Cols};
        }

        void fill(const value_type& value)
        {
            _fill(this->view(), value, this->device_type());
        }


        Mat<value_type> add(const Mat<value_type>& rhs) const;   
        Mat<value_type> operator+(const Mat<value_type>& rhs) const;             
        Mat<value_type> sub(const Mat<value_type>& rhs) const;  
        Mat<value_type> operator-(const Mat<value_type>& rhs) const;
        Mat<value_type> mul(const Mat<value_type>& rhs) const;    
        Mat<value_type> operator*(const Mat<value_type>& rhs) const;
        Mat<value_type> div(const Mat<value_type>& rhs) const;      
        Mat<value_type> operator/(const Mat<value_type>& rhs) const;

        void copyToHost(value_type* dest) const;
        void copyToDevice(value_type* dest) const;        

        Device device() const {return m_Device;}
        int rows() const {return m_Rows;}
        int cols() const {return m_Cols;}
        DEVICE_TYPE device_type() const {return m_Device.m_Dt;}
        std::string dtype(){return om::dtype<value_type>();}
        */
    private:

        void _compute_strides() 
        {
            m_Stride.resize(m_Shape.size());
            m_Stride.back() = 1;
            for (int i = m_Shape.size() - 2; i >= 0; --i) {
                m_Stride[i] = m_Stride[i + 1] * m_Shape[i + 1];
            }
        }

        std::vector<int> m_Shape;
        std::vector<int> m_Stride;
        _Ty* m_Data;
        Device m_Device;

        std::unique_ptr<Allocator<_Ty>> m_Allocator;
    };
}

#include "tensor.inl"
