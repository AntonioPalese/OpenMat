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
om::Tensor<value_type>::Tensor(Tensor &&rhs)
    : m_Shape(std::move(rhs.m_Shape)),
      m_Stride(std::move(rhs.m_Stride)),
      m_Device(rhs.m_Device),
      m_Data(rhs.m_Data),
      m_Allocator(std::move(rhs.m_Allocator))
{
    rhs.m_Data = nullptr;
}

template <typename value_type>
om::Tensor<value_type>::~Tensor()
{
    if(m_Data)
        m_Allocator->deallocate(m_Data);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::zeros(const std::vector<size_t>& shape, const Device& dv)
{
    Tensor<value_type> t(shape, dv);
    t.fill(static_cast<value_type>(0));
    return t;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::ones(const std::vector<size_t>& shape, const Device& dv)
{
    Tensor<value_type> t(shape, dv);
    t.fill(static_cast<value_type>(1));
    return t;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::full(const std::vector<size_t>& shape, value_type value, const Device& dv)
{
    Tensor<value_type> t(shape, dv);
    t.fill(value);
    return t;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::from_vector(const std::vector<value_type>& data,
                                                           const std::vector<size_t>& shape,
                                                           const Device& dv)
{
    size_t expected = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
    if (data.size() != expected)
        throw std::invalid_argument("from_vector: data size does not match shape");

    Tensor<value_type> t(shape, dv);
    if (dv.m_Dt == DEVICE_TYPE::CPU) {
        std::memcpy(t.m_Data, data.data(), sizeof(value_type) * expected);
    } else {
        CUDA_CALL(cudaMemcpy(t.m_Data, data.data(), sizeof(value_type) * expected, cudaMemcpyHostToDevice));
    }
    return t;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::add(const Tensor<value_type> &rhs) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _add(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator+(const Tensor<value_type> &rhs) const
{
    return this->add(rhs);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::sub(const Tensor<value_type> &rhs) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _sub(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator-(const Tensor<value_type> &rhs) const
{
    return this->sub(rhs);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::mul(const Tensor<value_type> &rhs) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _mul(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator*(const Tensor<value_type> &rhs) const
{
    return this->mul(rhs);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::div(const Tensor<value_type> &rhs) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _div(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator/(const Tensor<value_type> &rhs) const
{
    return this->div(rhs);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::matmul(const Tensor<value_type> &rhs) const
{
    // Validate 2D tensors
    if (this->rank() != 2 || rhs.rank() != 2) {
        throw std::runtime_error("matmul: both tensors must be 2D matrices");
    }

    // A is M×K, B is K×N, C is M×N
    size_t M = this->shape()[0];
    size_t K = this->shape()[1];
    size_t K2 = rhs.shape()[0];
    size_t N = rhs.shape()[1];

    if (K != K2) {
        throw std::runtime_error("matmul: inner dimensions must match (A.cols=" + 
            std::to_string(K) + " != B.rows=" + std::to_string(K2) + ")");
    }

    // Create output tensor with shape {M, N}
    Tensor<value_type> out({M, N}, this->device());

    _matmul(this->view(), rhs.view(), out.view(), this->device_type());

    return out;
}

/// 
template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::add(const value_type& scalar) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _add_k(this->view(), scalar, out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator+(const value_type& scalar) const
{
    return this->add(scalar);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::sub(const value_type& scalar) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _sub_k(this->view(), scalar, out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator-(const value_type& scalar) const
{
    return this->sub(scalar);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::mul(const value_type& scalar) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _mul_k(this->view(), scalar, out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator*(const value_type& scalar) const
{
    return this->mul(scalar);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::div(const value_type& scalar) const
{
    Tensor<value_type> out(this->shape(), this->device());

    _div_k(this->view(), scalar, out.view(), this->device_type());

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::operator/(const value_type& scalar) const
{
    return this->div(scalar);
}
////

template <typename value_type>
value_type om::Tensor<value_type>::sum() const
{
    if (device_type() == DEVICE_TYPE::CPU)
        return reduce_sum_cpu<value_type>(this->view());
    return launch_reduce_sum<value_type>(this->view());
}

template <typename value_type>
value_type om::Tensor<value_type>::mean() const
{
    value_type s = this->sum();
    size_t n = this->size();
    if (n == 0) throw std::invalid_argument("mean: empty tensor");
    return static_cast<value_type>(static_cast<double>(s) / static_cast<double>(n));
}

template <typename value_type>
value_type om::Tensor<value_type>::min() const
{
    if (device_type() == DEVICE_TYPE::CPU)
        return reduce_min_cpu<value_type>(this->view());
    return launch_reduce_min<value_type>(this->view());
}

template <typename value_type>
value_type om::Tensor<value_type>::max() const
{
    if (device_type() == DEVICE_TYPE::CPU)
        return reduce_max_cpu<value_type>(this->view());
    return launch_reduce_max<value_type>(this->view());
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::reshape(const std::vector<size_t>& new_shape) const
{
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>{});
    if (new_size != this->size())
        throw std::invalid_argument("reshape: new shape must have the same total number of elements");

    Tensor<value_type> out(*this);
    out.m_Shape = new_shape;
    out.m_Stride.resize(new_shape.size());
    out.m_Stride.back() = 1;
    for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i)
        out.m_Stride[i] = out.m_Stride[i + 1] * new_shape[i + 1];
    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::flatten() const
{
    return this->reshape({this->size()});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::squeeze(size_t axis) const
{
    if (axis >= this->rank())
        throw std::out_of_range("squeeze: axis out of range");
    if (m_Shape[axis] != 1)
        throw std::invalid_argument("squeeze: dimension at axis must be 1");

    std::vector<size_t> new_shape;
    new_shape.reserve(this->rank() - 1);
    for (size_t i = 0; i < this->rank(); ++i)
        if (i != axis) new_shape.push_back(m_Shape[i]);

    if (new_shape.empty()) new_shape.push_back(1);
    return this->reshape(new_shape);
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::unsqueeze(size_t axis) const
{
    if (axis > this->rank())
        throw std::out_of_range("unsqueeze: axis out of range");

    std::vector<size_t> new_shape;
    new_shape.reserve(this->rank() + 1);
    for (size_t i = 0; i < axis; ++i)
        new_shape.push_back(m_Shape[i]);
    new_shape.push_back(1);
    for (size_t i = axis; i < this->rank(); ++i)
        new_shape.push_back(m_Shape[i]);

    return this->reshape(new_shape);
}

template <typename value_type>
template <typename Op>
om::Tensor<value_type> om::Tensor<value_type>::apply(Op op) const
{
    Tensor<value_type> out(this->shape(), this->device());
    if (this->device_type() == DEVICE_TYPE::CPU) {
        auto src = this->view();
        auto dst = out.view();
        size_t n = src.size();
        for (size_t i = 0; i < n; ++i)
            dst[i] = op(src[i]);
    } else {
        launch_apply_op<value_type>(this->view(), out.view(), op);
    }
    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::scale_shift(value_type scale, value_type shift) const
{
    return this->apply(Compose<Mul<value_type>, Add<value_type>>{Mul<value_type>{scale}, Add<value_type>{shift}});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::shift_scale(value_type shift, value_type scale) const
{
    return this->apply(Compose<Add<value_type>, Mul<value_type>>{Add<value_type>{shift}, Mul<value_type>{scale}});
}

template <typename value_type>
template <typename Op>
om::Tensor<value_type> om::Tensor<value_type>::apply_binary(const Tensor<value_type>& rhs, Op op) const
{
    Tensor<value_type> out(this->shape(), this->device());
    if (this->device_type() == DEVICE_TYPE::CPU) {
        auto lhs_v = this->view();
        auto rhs_v = rhs.view();
        auto dst_v = out.view();
        if (!lhs_v.match(rhs_v))
            throw std::invalid_argument("apply_binary: tensors must have the same shape");
        size_t n = lhs_v.size();
        for (size_t i = 0; i < n; ++i)
            dst_v[i] = op(lhs_v[i], rhs_v[i]);
    } else {
        launch_apply_binary_op<value_type>(this->view(), rhs.view(), out.view(), op);
    }
    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::fused_add_mul(const Tensor<value_type>& rhs, value_type scale) const
{
    return this->apply_binary(rhs, BinaryCompose<BinaryAdd<value_type>, Mul<value_type>>{BinaryAdd<value_type>{}, Mul<value_type>{scale}});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::fused_sub_mul(const Tensor<value_type>& rhs, value_type scale) const
{
    return this->apply_binary(rhs, BinaryCompose<BinarySub<value_type>, Mul<value_type>>{BinarySub<value_type>{}, Mul<value_type>{scale}});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::fused_mul_add(const Tensor<value_type>& rhs, value_type shift) const
{
    return this->apply_binary(rhs, BinaryCompose<BinaryMul<value_type>, Add<value_type>>{BinaryMul<value_type>{}, Add<value_type>{shift}});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::fused_div_add(const Tensor<value_type>& rhs, value_type shift) const
{
    return this->apply_binary(rhs, BinaryCompose<BinaryDiv<value_type>, Add<value_type>>{BinaryDiv<value_type>{}, Add<value_type>{shift}});
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::to(const Device& target) const
{
    if (target.m_Dt == this->device_type()) {
        return Tensor<value_type>(*this);
    }

    Tensor<value_type> out(this->shape(), target);
    size_t n = this->size();

    if (this->device_type() == DEVICE_TYPE::CPU && target.m_Dt == DEVICE_TYPE::CUDA) {
        CUDA_CALL(cudaMemcpy(out.m_Data, m_Data, sizeof(value_type) * n, cudaMemcpyHostToDevice));
    } else {
        // CUDA → CPU
        CUDA_CALL(cudaMemcpy(out.m_Data, m_Data, sizeof(value_type) * n, cudaMemcpyDeviceToHost));
    }

    return out;
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::cpu() const
{
    return this->to(Device(0, DEVICE_TYPE::CPU));
}

template <typename value_type>
om::Tensor<value_type> om::Tensor<value_type>::cuda() const
{
    return this->to(Device(0, DEVICE_TYPE::CUDA));
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
