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
#include "storage.h"
#include "tensor_view.cuh"
#include "device_tensor_view.cuh"
#include "kernel_launcher.h"
#include "ops/kernels/fused_op.cuh"
#include "ops/kernels/transpose_gpu.cuh"
#include "ops/cpu/transpose_cpu.h"
#include "stream.h"


namespace om
{
    template<typename _Ty>
    class Tensor
    {
    public:
        using value_type = _Ty;

        Tensor(const std::vector<size_t>& shape, const Device& dv = Device(0, DEVICE_TYPE::CPU));
        // Copy is a deep copy into a fresh contiguous buffer, never a view:
        // views come only from the methods that say so (reshape, slice, …).
        Tensor(const Tensor& rhs);
        Tensor(Tensor&& rhs);
        Tensor& operator=(Tensor&& rhs);
        ~Tensor() = default;

        static Tensor<value_type> zeros(const std::vector<size_t>& shape,
                                        const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> ones(const std::vector<size_t>& shape,
                                       const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> full(const std::vector<size_t>& shape, value_type value,
                                       const Device& dv = Device(0, DEVICE_TYPE::CPU));
        static Tensor<value_type> from_vector(const std::vector<value_type>& data,
                                              const std::vector<size_t>& shape,
                                              const Device& dv = Device(0, DEVICE_TYPE::CPU));

        // A CPU tensor backed by page-locked (cudaHostAlloc) memory instead of
        // HostPool's ordinary pageable blocks. Use it for a host tensor you
        // know will repeatedly cross the bus as the *source* of an H2D
        // transfer (Tensor::to()/cuda()) — pinning turns that cudaMemcpyAsync
        // into a genuine DMA instead of one staged through the driver's own
        // pinned bounce buffer. The destination side of a D2H transfer is
        // pinned automatically by Tensor::to(); there is no equivalent
        // automatic case on the H2D side because to() cannot retroactively
        // pin a source tensor that already exists — see host_pool.h.
        static Tensor<value_type> pinned(const std::vector<size_t>& shape);

        // True iff this is a CPU tensor allocated via Tensor::pinned() or as
        // the destination of a device-to-host Tensor::to().
        bool is_pinned() const;

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
        

        // Kept for source compatibility; fill_ is the canonical spelling and
        // the one that takes a stream.
        void fill(const value_type& value) { this->fill_(value); }
        
        
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

        // ── Views ───────────────────────────────────────────────────────────
        //
        // A view shares this tensor's Storage: a write through either one is
        // visible through the other, and the buffer lives until the last of
        // them is destroyed. reshape/flatten return a view when this tensor is
        // contiguous and a reshaped copy otherwise (PyTorch's rule);
        // squeeze, unsqueeze, slice and select always return a view.

        Tensor<value_type> reshape(const std::vector<size_t>& new_shape) const;
        Tensor<value_type> flatten() const;
        Tensor<value_type> squeeze(size_t axis) const;
        Tensor<value_type> unsqueeze(size_t axis) const;

        // Elements start, start+step, … below stop along `axis`. step > 0.
        Tensor<value_type> slice(size_t axis, size_t start, size_t stop, size_t step = 1) const;
        // Fixes `axis` at `index` and drops it. On a rank-1 tensor the result
        // has shape {1}, the same convention as squeeze.
        Tensor<value_type> select(size_t axis, size_t index) const;

        // A view of this tensor when it is already contiguous, a contiguous
        // copy otherwise.
        Tensor<value_type> contiguous() const;
        // Always a contiguous copy; the same as the copy constructor.
        Tensor<value_type> clone() const { return Tensor<value_type>(*this); }

        bool is_contiguous() const { return this->view().is_contiguous(); }
        size_t storage_offset() const { return m_Offset; }
        bool shares_storage(const Tensor<value_type>& other) const
        { return m_Storage && m_Storage == other.m_Storage; }

        // Writes `src` into this tensor (which may be a view), broadcasting
        // src to this tensor's shape. Same device only.
        Tensor<value_type>& copy_(const Tensor<value_type>& src, const Stream& s);
        Tensor<value_type>& copy_(const Tensor<value_type>& src);

        Tensor<value_type> transpose() const;
        Tensor<value_type> permute(const std::vector<size_t>& axes) const;

        // ── Stream overloads ────────────────────────────────────────────────
        // All return a new Tensor; the caller is responsible for synchronizing
        // the stream before reading results.

        Tensor<value_type> add(const Tensor<value_type>& rhs, const Stream& s) const;
        Tensor<value_type> sub(const Tensor<value_type>& rhs, const Stream& s) const;
        Tensor<value_type> mul(const Tensor<value_type>& rhs, const Stream& s) const;
        Tensor<value_type> div(const Tensor<value_type>& rhs, const Stream& s) const;

        Tensor<value_type> add(const value_type& scalar, const Stream& s) const;
        Tensor<value_type> sub(const value_type& scalar, const Stream& s) const;
        Tensor<value_type> mul(const value_type& scalar, const Stream& s) const;
        Tensor<value_type> div(const value_type& scalar, const Stream& s) const;

        Tensor<value_type> matmul(const Tensor<value_type>& rhs, const Stream& s) const;

        Tensor<value_type> transpose(const Stream& s) const;
        Tensor<value_type> permute(const std::vector<size_t>& axes, const Stream& s) const;

        template<typename Op>
        Tensor<value_type> apply(Op op, const Stream& s) const;

        Tensor<value_type> relu(const Stream& s) const;
        Tensor<value_type> sigmoid(const Stream& s) const;

        template<typename Op>
        Tensor<value_type> apply(Op op) const;

        Tensor<value_type> scale_shift(value_type scale, value_type shift) const;
        Tensor<value_type> shift_scale(value_type shift, value_type scale) const;

        template<typename Op>
        Tensor<value_type> apply_binary(const Tensor<value_type>& rhs, Op op, const Stream& s) const;

        template<typename Op>
        Tensor<value_type> apply_binary(const Tensor<value_type>& rhs, Op op) const;


        Tensor<value_type> relu() const;
        Tensor<value_type> sigmoid() const;

        Tensor<value_type> fused_add_mul(const Tensor<value_type>& rhs, value_type scale) const;

        Tensor<value_type> fused_sub_mul(const Tensor<value_type>& rhs, value_type scale) const;

        Tensor<value_type> fused_mul_add(const Tensor<value_type>& rhs, value_type shift) const;
    
        Tensor<value_type> fused_div_add(const Tensor<value_type>& rhs, value_type shift) const;

        // ── Destination-provided overloads ──────────────────────────────────
        //
        // `a.add_out(b, out)` writes a+b into `out`'s existing buffer and
        // returns a reference to it, so a loop running the same op every
        // iteration allocates once instead of once per iteration. `out` must
        // already carry the result's shape and sit on the same device; it may
        // be one of the operands (that is what the in-place family below is).
        //
        // These are the single implementation of each op: the allocating forms
        // above build the result and call straight into them, and the in-place
        // forms pass *this as the destination. Nothing else branches on
        // device_type().

        Tensor<value_type>& add_out(const Tensor<value_type>& rhs, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& add_out(const Tensor<value_type>& rhs, Tensor<value_type>& out) const;
        Tensor<value_type>& sub_out(const Tensor<value_type>& rhs, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& sub_out(const Tensor<value_type>& rhs, Tensor<value_type>& out) const;
        Tensor<value_type>& mul_out(const Tensor<value_type>& rhs, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& mul_out(const Tensor<value_type>& rhs, Tensor<value_type>& out) const;
        Tensor<value_type>& div_out(const Tensor<value_type>& rhs, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& div_out(const Tensor<value_type>& rhs, Tensor<value_type>& out) const;

        Tensor<value_type>& add_out(const value_type& scalar, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& add_out(const value_type& scalar, Tensor<value_type>& out) const;
        Tensor<value_type>& sub_out(const value_type& scalar, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& sub_out(const value_type& scalar, Tensor<value_type>& out) const;
        Tensor<value_type>& mul_out(const value_type& scalar, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& mul_out(const value_type& scalar, Tensor<value_type>& out) const;
        Tensor<value_type>& div_out(const value_type& scalar, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& div_out(const value_type& scalar, Tensor<value_type>& out) const;

        // matmul, transpose and permute each read an index they do not write,
        // so unlike the elementwise family `out` may not be an operand.
        Tensor<value_type>& matmul_out(const Tensor<value_type>& rhs, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& matmul_out(const Tensor<value_type>& rhs, Tensor<value_type>& out) const;
        Tensor<value_type>& transpose_out(Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& transpose_out(Tensor<value_type>& out) const;
        Tensor<value_type>& permute_out(const std::vector<size_t>& axes, Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& permute_out(const std::vector<size_t>& axes, Tensor<value_type>& out) const;

        template<typename Op>
        Tensor<value_type>& apply_out(Op op, Tensor<value_type>& out, const Stream& s) const;
        template<typename Op>
        Tensor<value_type>& apply_out(Op op, Tensor<value_type>& out) const;
        template<typename Op>
        Tensor<value_type>& apply_binary_out(const Tensor<value_type>& rhs, Op op,
                                             Tensor<value_type>& out, const Stream& s) const;
        template<typename Op>
        Tensor<value_type>& apply_binary_out(const Tensor<value_type>& rhs, Op op,
                                             Tensor<value_type>& out) const;

        Tensor<value_type>& relu_out(Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& relu_out(Tensor<value_type>& out) const;
        Tensor<value_type>& sigmoid_out(Tensor<value_type>& out, const Stream& s) const;
        Tensor<value_type>& sigmoid_out(Tensor<value_type>& out) const;

        // ── In-place ops ────────────────────────────────────────────────────
        //
        // Trailing underscore, as in PyTorch: the result replaces this
        // tensor's contents, no allocation and no free happen, and the return
        // is *this so calls chain. Every one of them is `X_out(..., *this)`.
        //
        // They are only defined for ops whose kernels read and write the same
        // index — the elementwise families, `apply`, `apply_binary` and
        // `fill`. matmul, transpose and permute have no in-place form.

        Tensor<value_type>& add_(const Tensor<value_type>& rhs, const Stream& s);
        Tensor<value_type>& add_(const Tensor<value_type>& rhs);
        Tensor<value_type>& sub_(const Tensor<value_type>& rhs, const Stream& s);
        Tensor<value_type>& sub_(const Tensor<value_type>& rhs);
        Tensor<value_type>& mul_(const Tensor<value_type>& rhs, const Stream& s);
        Tensor<value_type>& mul_(const Tensor<value_type>& rhs);
        Tensor<value_type>& div_(const Tensor<value_type>& rhs, const Stream& s);
        Tensor<value_type>& div_(const Tensor<value_type>& rhs);

        Tensor<value_type>& add_(const value_type& scalar, const Stream& s);
        Tensor<value_type>& add_(const value_type& scalar);
        Tensor<value_type>& sub_(const value_type& scalar, const Stream& s);
        Tensor<value_type>& sub_(const value_type& scalar);
        Tensor<value_type>& mul_(const value_type& scalar, const Stream& s);
        Tensor<value_type>& mul_(const value_type& scalar);
        Tensor<value_type>& div_(const value_type& scalar, const Stream& s);
        Tensor<value_type>& div_(const value_type& scalar);

        Tensor<value_type>& operator+=(const Tensor<value_type>& rhs);
        Tensor<value_type>& operator-=(const Tensor<value_type>& rhs);
        Tensor<value_type>& operator*=(const Tensor<value_type>& rhs);
        Tensor<value_type>& operator/=(const Tensor<value_type>& rhs);
        Tensor<value_type>& operator+=(const value_type& scalar);
        Tensor<value_type>& operator-=(const value_type& scalar);
        Tensor<value_type>& operator*=(const value_type& scalar);
        Tensor<value_type>& operator/=(const value_type& scalar);

        template<typename Op>
        Tensor<value_type>& apply_(Op op, const Stream& s);
        template<typename Op>
        Tensor<value_type>& apply_(Op op);
        template<typename Op>
        Tensor<value_type>& apply_binary_(const Tensor<value_type>& rhs, Op op, const Stream& s);
        template<typename Op>
        Tensor<value_type>& apply_binary_(const Tensor<value_type>& rhs, Op op);

        Tensor<value_type>& relu_(const Stream& s);
        Tensor<value_type>& relu_();
        Tensor<value_type>& sigmoid_(const Stream& s);
        Tensor<value_type>& sigmoid_();

        Tensor<value_type>& fill_(const value_type& value, const Stream& s);
        Tensor<value_type>& fill_(const value_type& value);

        Tensor<value_type> to(const Device& target) const;
        Tensor<value_type> cpu() const;
        Tensor<value_type> cuda() const;

        // Async transfer overloads — caller must synchronize the stream before reading.
        Tensor<value_type> to(const Device& target, const Stream& s) const;
        Tensor<value_type> cpu(const Stream& s) const;
        Tensor<value_type> cuda(const Stream& s) const;

        static Tensor<value_type> from_vector(const std::vector<value_type>& data,
                                              const std::vector<size_t>& shape,
                                              const Device& dv,
                                              const Stream& s);

        void copyToHost(value_type* dest) const;
        void copyToDevice(value_type* dest) const;

        Device device() const {return m_Device;}

        std::vector<size_t> shape() const {return m_Shape;}
        const size_t* shape_p() const {return m_Shape.data();}
        std::vector<size_t> stride() const {return m_Stride;}
        const size_t* stride_p() const {return m_Stride.data();}

        DEVICE_TYPE device_type() const {return m_Device.m_Dt;}
        std::string dtype() const {return om::dtype<value_type>();}
        size_t size() const {return std::accumulate(m_Shape.begin(), m_Shape.end(), size_t{1}, std::multiplies<>());}
        size_t rank() const {return m_Shape.size();}
        // The stream the underlying buffer was allocated on and will be freed
        // on. Every view of a tensor reports its storage's stream.
        const Stream& stream() const;

    private:
        // A view: shares `storage`, starts `offset` elements into it.
        Tensor(std::shared_ptr<Storage<_Ty>> storage, size_t offset,
               std::vector<size_t> shape, std::vector<size_t> stride);

        // A view of this tensor with a new shape and stride over the same
        // elements' start.
        Tensor<value_type> _alias(std::vector<size_t> shape, std::vector<size_t> stride,
                                  size_t extra_offset = 0) const;

        // Fills `out` (already allocated, same shape) with this tensor's
        // elements, whatever the strides of the two. The CPU loop or the GPU
        // strided copy kernel; the one place that copies a view.
        void _copy_into(Tensor<value_type>& out, const Stream& s) const;

        // Internal constructor used by stream overloads to associate output
        // tensors with the enqueuing stream for async alloc/free.
        Tensor(const std::vector<size_t>& shape, const Device& dv, Stream stream);

        // Same, plus an explicit choice of PinnedCpuAllocator over the
        // device-default allocator. `dv` must be CPU when `pinned` is true —
        // there is no such thing as pinned device memory.
        Tensor(const std::vector<size_t>& shape, const Device& dv, Stream stream, bool pinned);

        // Throws unless `out` can receive a result of shape `shape` produced on
        // this tensor's device. `who` names the caller in the message.
        void _check_out(const Tensor<value_type>& out,
                        const std::vector<size_t>& shape,
                        const char* who) const;

        void _check_out_contiguous(const Tensor<value_type>& out, const char* who) const;

        // Throws unless `rhs` is a legal second operand for an elementwise op:
        // same device, and a shape that broadcasts against this one (NumPy
        // rules). Returns the broadcast result shape.
        std::vector<size_t> _check_operand(const Tensor<value_type>& rhs, const char* who) const;

        // Every elementwise path — the CPU loop, the contiguous GPU fast path
        // and the rank-specialized kernels — reads index i and writes index i
        // through each operand's own strides. So an operand that is *the same
        // view* as the destination (same start, shape and strides — the
        // in-place case) is exactly as correct as a separate buffer. Any other
        // overlap between an operand and the destination — a shifted slice of
        // the same storage, a broadcast operand that is also written — would
        // read elements already overwritten, so it throws. `lhs`/`rhs` are the
        // operands as the kernel will see them, i.e. already expanded to the
        // result shape.
        void _check_alias_elementwise(const Tensor<value_type>& out,
                                      const TensorView<const value_type>& lhs,
                                      const Tensor<value_type>* rhs,
                                      const TensorView<const value_type>* rhs_view,
                                      const char* who) const;

        // For ops that read an index they do not write (matmul, transpose,
        // permute), where any overlap with the destination is wrong.
        void _check_alias_none(const Tensor<value_type>& out,
                               const Tensor<value_type>* rhs,
                               const char* who) const;

        void _compute_strides();
        inline size_t _compute_flat_index(const std::vector<size_t>& indices) const;

        std::vector<size_t> m_Shape;
        std::vector<size_t> m_Stride;
        // m_Storage->data() + m_Offset: where this tensor's element (0,…,0)
        // lives. Cached because every kernel launch needs it.
        _Ty* m_Data = nullptr;
        Device m_Device;
        size_t m_Offset = 0;

        std::shared_ptr<Storage<_Ty>> m_Storage;
    };
}

#include "tensor.inl"
