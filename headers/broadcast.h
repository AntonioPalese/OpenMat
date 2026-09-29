#pragma once
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor_view.cuh"
#include "device_tensor_view.cuh"

// NumPy-style broadcasting, done entirely on the host.
//
// Nothing here touches data. An operand is broadcast by handing the kernel a
// view whose shape is the *output* shape and whose stride is 0 on every axis
// the operand does not really have (a missing leading axis, or an axis of
// extent 1 stretched to n). Every kernel that indexes through strides — the
// rank-specialized GPU kernels, the _nd kernels, the strided CPU loop — then
// reads the same element for every coordinate along that axis, with no copy.
//
// When the two shapes are already equal, expand_to returns the operand's own
// strides unchanged, so the views are exactly what they were before
// broadcasting existed and the contiguous fast paths still apply. A view with
// a 0 stride on an axis of extent > 1 is not contiguous, so those fast paths
// decline it on their own and the stride-aware kernels take over.
namespace om::detail
{
    inline std::string shape_str(const std::vector<size_t>& s)
    {
        std::string r = "(";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) r += ", ";
            r += std::to_string(s[i]);
        }
        return r + ")";
    }

    // Right-align the shapes; per axis the extents must be equal or one of
    // them 1, and the result takes the larger. A missing leading axis counts
    // as 1.
    inline std::vector<size_t> broadcast_shapes(const std::vector<size_t>& a,
                                                const std::vector<size_t>& b,
                                                const char* who)
    {
        const size_t rank = a.size() > b.size() ? a.size() : b.size();
        // Checked here rather than left to the DeviceTensorView assert, which
        // Release builds compile out.
        if (rank > MAX_RANK)
            throw std::invalid_argument(std::string(who) + ": result rank " +
                std::to_string(rank) + " exceeds MAX_RANK (8)");

        std::vector<size_t> out(rank);
        for (size_t i = 0; i < rank; ++i) {
            const size_t ea = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
            const size_t eb = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
            if (ea != eb && ea != 1 && eb != 1)
                throw std::invalid_argument(std::string(who) + ": shapes " +
                    shape_str(a) + " and " + shape_str(b) + " are not broadcastable");
            out[i] = ea == 1 ? eb : ea;
        }
        return out;
    }

    // An operand's layout stretched to the output shape. The arrays are
    // inline, so there is no allocation per op and view() can point into
    // them: the layout is a local that outlives the launch, and the launch
    // copies shape/stride by value into DeviceTensorView before returning.
    struct ExpandedLayout
    {
        size_t shape[MAX_RANK];
        size_t stride[MAX_RANK];
        size_t rank;

        template <typename T>
        TensorView<const T> view(const T* data) const
        {
            return TensorView<const T>{data, shape, stride, rank};
        }
    };

    // The span of memory a view can touch: from its first element to one past
    // its last, [data, data + Σ(shape-1)·stride]. Strides are unsigned, so
    // the first element is always the lowest address. An empty view touches
    // nothing.
    template <typename T>
    inline bool views_overlap(const TensorView<const T>& a, const TensorView<const T>& b)
    {
        auto last = [](const TensorView<const T>& v) {
            size_t off = 0;
            for (size_t i = 0; i < v.rank; ++i) {
                if (v.shape[i] == 0) return static_cast<const T*>(nullptr);
                off += (v.shape[i] - 1) * v.stride[i];
            }
            return v.data + off;
        };
        const T* a_last = last(a);
        const T* b_last = last(b);
        if (!a_last || !b_last) return false;
        return a.data <= b_last && b.data <= a_last;
    }

    // Same start, shape and strides: every index names the same element in
    // both, which is the one overlap an elementwise op can survive.
    template <typename T>
    inline bool same_layout(const TensorView<const T>& a, const TensorView<const T>& b)
    {
        if (a.data != b.data || !a.same_shape(b)) return false;
        for (size_t i = 0; i < a.rank; ++i)
            if (a.shape[i] != 1 && a.stride[i] != b.stride[i]) return false;
        return true;
    }

    // `out_shape` must come from broadcast_shapes with this shape as one side.
    inline ExpandedLayout expand_to(const std::vector<size_t>& shape,
                                    const std::vector<size_t>& stride,
                                    const std::vector<size_t>& out_shape)
    {
        ExpandedLayout l{};
        l.rank = out_shape.size();
        const size_t lead = l.rank - shape.size();
        for (size_t i = 0; i < l.rank; ++i) {
            l.shape[i] = out_shape[i];
            if (i < lead)
                l.stride[i] = 0;
            else if (shape[i - lead] == 1 && out_shape[i] != 1)
                l.stride[i] = 0;
            else
                l.stride[i] = stride[i - lead];
        }
        return l;
    }
}
