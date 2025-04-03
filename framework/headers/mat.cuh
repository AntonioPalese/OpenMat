#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <memory>

#include "mat_utils.h"
#include "allocator.h"
#include "mat_view.cuh"
#include "kernel_launcher.h"


namespace om
{
    template<typename _Ty>
    class Mat
    {
    public:
        using value_type = _Ty;

        Mat(int r, int c, const Device& dv = Device(0, DEVICE_TYPE::CPU));
        Mat(const Mat& rhs); // copy
        Mat(Mat&& rhs); // move
        ~Mat();

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

        Mat<value_type> add(Mat<value_type>& rhs) 
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
        

        void copyToHost(value_type* dest) const;
        void copyToDevice(value_type* dest) const;        

        Device device() const {return m_Device;}
        int rows() const {return m_Rows;}
        int cols() const {return m_Cols;}
        DEVICE_TYPE device_type() const {return m_Device.m_Dt;}

    private:
        int m_Rows;
        int m_Cols;
        _Ty* m_Data;
        Device m_Device;

        std::unique_ptr<Allocator<_Ty>> m_Allocator;
    };
}

#include "mat.inl"
