#include "test_helpers.h"
#include <numeric>
#include <string>

// The contiguous fast path (headers/ops/kernels/contiguous.cuh) replaces the
// rank-specialized kernels whenever the buffers are one flat row-major run,
// which is every tensor the library builds today. Two things about it are not
// covered by the per-op suites:
//
//  * the packed variant. For char and float16_t a thread handles 4 and 2
//    elements at once, punned through a 4-byte word, and a size that is not a
//    multiple of that width leaves a tail one block has to pick up. Nothing
//    else in the suite runs a sub-4-byte dtype through the GPU at a size like
//    4095, so nothing else would notice the tail being dropped or written twice.
//
//  * rank independence. The fast path throws the shape away, so the same
//    element count laid out as a vector, a matrix or a 5-D tensor must give bit
//    identical results — that is the invariant that lets it skip the
//    rank-specialized kernels at all.

namespace {

// Sizes chosen so n % 4 covers 0, 1, 2 and 3: the packed path's tail is the
// only part of the kernel a size-oblivious test would never reach.
const std::vector<size_t> kSizes = {1, 2, 3, 4, 5, 7, 8, 255, 256, 257, 1023, 4095, 65537};

template <typename T>
std::vector<T> to_host_vec(const Tensor<T>& t) {
    std::vector<T> v(t.size());
    t.copyToHost(v.data());
    return v;
}

// Small integers, exactly representable in char and float16_t alike, so every
// dtype can be compared for equality rather than tolerance.
template <typename T>
Tensor<T> make_host(const std::vector<size_t>& shape, int base) {
    Device cpu("cpu:0");
    Tensor<T> t(shape, cpu);
    size_t n = t.size();
    for (size_t i = 0; i < n; ++i)
        t.view()[i] = static_cast<T>(static_cast<int>(base + (i % 7)));
    return t;
}

template <typename T>
void expect_vec_eq(const std::vector<T>& got, const std::vector<T>& want, const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i]))
            << what << " at index " << i << " of " << got.size();
}

// Runs the four binary ops and the four tensor-scalar ops at one size, on both
// backends, and requires the GPU (fast path) to agree with the CPU loop.
template <typename T>
void check_size(size_t n) {
    Device gpu("cuda:0");

    Tensor<T> ha = make_host<T>({n}, 1);
    Tensor<T> hb = make_host<T>({n}, 2);
    Tensor<T> da = ha.to(gpu);
    Tensor<T> db = hb.to(gpu);

    const std::string tag = " n=" + std::to_string(n);
    expect_vec_eq(to_host_vec(da + db), to_host_vec(ha + hb), "add" + tag);
    expect_vec_eq(to_host_vec(da - db), to_host_vec(ha - hb), "sub" + tag);
    expect_vec_eq(to_host_vec(da * db), to_host_vec(ha * hb), "mul" + tag);
    expect_vec_eq(to_host_vec(da / db), to_host_vec(ha / hb), "div" + tag);

    const T k = static_cast<T>(3);
    expect_vec_eq(to_host_vec(da + k), to_host_vec(ha + k), "add_k" + tag);
    expect_vec_eq(to_host_vec(da - k), to_host_vec(ha - k), "sub_k" + tag);
    expect_vec_eq(to_host_vec(da * k), to_host_vec(ha * k), "mul_k" + tag);
    expect_vec_eq(to_host_vec(da / k), to_host_vec(ha / k), "div_k" + tag);
}

} // namespace

TEST(Contiguous, BinaryAndScalarOpsFloat) {
    OM_REQUIRE_CUDA();
    for (size_t n : kSizes) check_size<float>(n);
}

TEST(Contiguous, BinaryAndScalarOpsInt) {
    OM_REQUIRE_CUDA();
    for (size_t n : kSizes) check_size<int>(n);
}

// char packs 4 elements per thread — the widest pack, so the most tail cases.
TEST(Contiguous, BinaryAndScalarOpsChar) {
    OM_REQUIRE_CUDA();
    for (size_t n : kSizes) check_size<char>(n);
}

// float16_t packs 2 per thread, and is punned through a 4-byte word rather than
// copied member-wise, so a wrong pun shows up here and nowhere else.
TEST(Contiguous, BinaryAndScalarOpsHalf) {
    OM_REQUIRE_CUDA();
    for (size_t n : kSizes) check_size<float16_t>(n);
}

// The fast path ignores the shape, so every layout of the same element count
// must produce the same buffer. 5-D also exercises the case that used to fall
// through to the flat _nd kernel.
TEST(Contiguous, ResultIsIndependentOfRank) {
    OM_REQUIRE_CUDA();
    Device gpu("cuda:0");

    const std::vector<std::vector<size_t>> shapes = {
        {2 * 3 * 4 * 5 * 6}, {6, 120}, {4, 6, 30}, {2, 3, 4, 30}, {2, 3, 4, 5, 6},
    };

    std::vector<float> reference;
    for (const auto& shape : shapes) {
        Tensor<float> a = make_host<float>(shape, 1).to(gpu);
        Tensor<float> b = make_host<float>(shape, 2).to(gpu);
        std::vector<float> got = to_host_vec(a + b);
        if (reference.empty()) reference = got;
        else expect_vec_eq(got, reference, "rank " + std::to_string(shape.size()));
    }
}

// fill writes without reading, and takes the same packed path.
TEST(Contiguous, FillCoversTail) {
    OM_REQUIRE_CUDA();
    Device gpu("cuda:0");

    for (size_t n : kSizes) {
        Tensor<char> t({n}, gpu);
        t.fill(static_cast<char>(9));
        std::vector<char> got = to_host_vec(t);
        for (size_t i = 0; i < n; ++i)
            ASSERT_EQ(static_cast<int>(got[i]), 9) << "n=" << n << " index " << i;
    }
}

// The fused launchers take the fast path too, with an arbitrary functor rather
// than one of the four generated ops.
TEST(Contiguous, FusedOpsMatchHost) {
    OM_REQUIRE_CUDA();
    Device gpu("cuda:0");

    for (size_t n : {size_t{7}, size_t{4095}, size_t{65537}}) {
        Tensor<float> ha = make_host<float>({n}, -3);
        Tensor<float> hb = make_host<float>({n}, 2);
        Tensor<float> da = ha.to(gpu);
        Tensor<float> db = hb.to(gpu);

        const std::string tag = " n=" + std::to_string(n);
        expect_vec_eq(to_host_vec(da.relu()), to_host_vec(ha.relu()), "relu" + tag);
        expect_vec_eq(to_host_vec(da.fused_add_mul(db, 2.5f)),
                      to_host_vec(ha.fused_add_mul(hb, 2.5f)), "fused_add_mul" + tag);
        expect_vec_eq(to_host_vec(da.scale_shift(2.0f, 1.0f)),
                      to_host_vec(ha.scale_shift(2.0f, 1.0f)), "scale_shift" + tag);
    }
}

// A view whose strides are not row-major packed must NOT take the fast path.
// Nothing produces one today, so this pins the guard itself rather than a
// reachable code path: it is what keeps the flattening honest once views land.
TEST(Contiguous, IsContiguousRejectsStridedView) {
    Device cpu("cpu:0");
    Tensor<float> t({4, 6}, cpu);
    EXPECT_TRUE(t.view().is_contiguous());

    const size_t shape[2]  = {4, 6};
    const size_t packed[2] = {6, 1};
    const size_t strided[2] = {12, 1};   // every other row
    const size_t transposed[2] = {1, 4};

    EXPECT_TRUE((TensorView<float>{t.view().data, shape, packed, 2}.is_contiguous()));
    EXPECT_FALSE((TensorView<float>{t.view().data, shape, strided, 2}.is_contiguous()));
    EXPECT_FALSE((TensorView<float>{t.view().data, shape, transposed, 2}.is_contiguous()));

    // An axis of extent 1 carries no elements, so its stride is irrelevant.
    const size_t shape_1[3]  = {4, 1, 6};
    const size_t stride_1[3] = {6, 999, 1};
    EXPECT_TRUE((TensorView<float>{t.view().data, shape_1, stride_1, 3}.is_contiguous()));
}
