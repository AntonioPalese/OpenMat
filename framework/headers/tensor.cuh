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
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
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
        
        const value_type& operator()(std::initializer_list<size_t> indices) const
        {
            return m_Data[_compute_flat_index(indices)];
        }
        value_type& operator()(std::initializer_list<size_t> indices)
        {
            return m_Data[_compute_flat_index(indices)];
        }
        
        __host__ TensorView<value_type> view() 
        {
            return TensorView<value_type>{
                m_Data,
                m_Shape.data(),
                m_Stride.data(),
                m_Shape.size()
            };
        }
        
        __host__ TensorView<const value_type> view() const 
        {
            return TensorView<const value_type>{
                m_Data,
                m_Shape.data(),
                m_Stride.data(),
                m_Shape.size()
            };
        }
        

        void fill(const value_type& value)
        {
            _fill(this->view(), value, this->device_type());
        }
        
        /*
        Mat<value_type> add(const Mat<value_type>& rhs) const;   
        Mat<value_type> operator+(const Mat<value_type>& rhs) const;             
        Mat<value_type> sub(const Mat<value_type>& rhs) const;  
        Mat<value_type> operator-(const Mat<value_type>& rhs) const;
        Mat<value_type> mul(const Mat<value_type>& rhs) const;    
        Mat<value_type> operator*(const Mat<value_type>& rhs) const;
        Mat<value_type> div(const Mat<value_type>& rhs) const;      
        Mat<value_type> operator/(const Mat<value_type>& rhs) const;
        */
        void copyToHost(value_type* dest) const;
        void copyToDevice(value_type* dest) const;        

        Device device() const {return m_Device;}

        std::vector<size_t> shape() const {return m_Shape;}
        const size_t* shape_p() const {return m_Shape.data();}
        std::vector<size_t> stride() const {return m_Stride;}
        const size_t* stride_p() const {return m_Stride.data();}

        DEVICE_TYPE device_type() const {return m_Device.m_Dt;}
        std::string dtype() const {return om::dtype<value_type>();}
        size_t size() const {return std::accumulate(m_Shape.begin(), m_Shape.end(), 1, std::multiplies<>());}
        size_t rank() const {return m_Shape.size();}
    private:

        void _compute_strides();
        inline size_t _compute_flat_index(const std::vector<size_t>& indices) const;

        std::vector<size_t> m_Shape;
        std::vector<size_t> m_Stride;
        _Ty* m_Data;
        Device m_Device;

        std::unique_ptr<Allocator<_Ty>> m_Allocator;
    };
}

#include "tensor.inl"
