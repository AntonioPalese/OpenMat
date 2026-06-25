#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <memory>
#include <vector>
#include <utility>

#include "mat_utils.h"
#include "allocator.h"
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include "kernel_launcher.h"
#include "ops/kernels/fused_op.cuh"


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

        static Tensor<value_type> zeros(const std::vector<size_t>& shape,
                                        const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> ones(const std::vector<size_t>& shape,
                                       const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> full(const std::vector<size_t>& shape, value_type value,
                                       const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> from_vector(const std::vector<value_type>& data,
                                              const std::vector<size_t>& shape,
                                              const Device& dv = Device(0, DEVICE_TYPE::CPU));
        
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
        
        
        Tensor<value_type> add(const Tensor<value_type>& rhs) const;   
        Tensor<value_type> operator+(const Tensor<value_type>& rhs) const;  
        Tensor<value_type> sub(const Tensor<value_type>& rhs) const;  
        Tensor<value_type> operator-(const Tensor<value_type>& rhs) const;
        Tensor<value_type> mul(const Tensor<value_type>& rhs) const;    
        Tensor<value_type> operator*(const Tensor<value_type>& rhs) const;
        Tensor<value_type> div(const Tensor<value_type>& rhs) const;      
        Tensor<value_type> operator/(const Tensor<value_type>& rhs) const;
        
        Tensor<value_type> matmul(const Tensor<value_type>& rhs) const;

        Tensor<value_type> add(const value_type& scalar) const;   
        Tensor<value_type> operator+(const value_type& scalar) const;  
        Tensor<value_type> sub(const value_type& scalar) const;  
        Tensor<value_type> operator-(const value_type& scalar) const;
        Tensor<value_type> mul(const value_type& scalar) const;    
        Tensor<value_type> operator*(const value_type& scalar) const;
        Tensor<value_type> div(const value_type& scalar) const;      
        Tensor<value_type> operator/(const value_type& scalar) const;


        value_type sum() const;
        value_type mean() const;
        value_type min() const;
        value_type max() const;

        Tensor<value_type> reshape(const std::vector<size_t>& new_shape) const;
        Tensor<value_type> flatten() const;
        Tensor<value_type> squeeze(size_t axis) const;
        Tensor<value_type> unsqueeze(size_t axis) const;

        template<typename Op>
        Tensor<value_type> apply(Op op) const;

        Tensor<value_type> scale_shift(value_type scale, value_type shift) const;
        Tensor<value_type> shift_scale(value_type shift, value_type scale) const;

        template<typename Op>
        Tensor<value_type> apply_binary(const Tensor<value_type>& rhs, Op op) const;


        Tensor<value_type> relu() const;
        Tensor<value_type> sigmoid() const;

        Tensor<value_type> fused_add_mul(const Tensor<value_type>& rhs, value_type scale) const;

        Tensor<value_type> fused_sub_mul(const Tensor<value_type>& rhs, value_type scale) const;

        Tensor<value_type> fused_mul_add(const Tensor<value_type>& rhs, value_type shift) const;
    
        Tensor<value_type> fused_div_add(const Tensor<value_type>& rhs, value_type shift) const;

        Tensor<value_type> to(const Device& target) const;
        Tensor<value_type> cpu() const;
        Tensor<value_type> cuda() const;

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
