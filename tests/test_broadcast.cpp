// NumPy-style broadcasting for the binary elementwise ops.
//
// Expected values come from reference_binary() below, which walks the output
// with its own right-aligned index arithmetic on plain std::vectors — never
// through the library — so a stride bug in expand_to, the CPU loop or a kernel
// cannot also be present in the reference.
//
// The shape table is chosen to reach every code path a broadcast operand can
// take: the rank-1..4 specialized kernels, the _nd kernel at rank 5 and 6, and
// the _nd fallback a rank-4 launch takes when shape[0] overflows gridDim.z.
#include "test_helpers.h"
#include "broadcast.h"
#include <functional>
#include <numeric>
#include <string>

namespace {

size_t numel(const std::vector<size_t>& s) {
    return std::accumulate(s.begin(), s.end(), size_t{1}, std::multiplies<>());
}

std::string str(const std::vector<size_t>& s) { return detail::shape_str(s); }

// Small integers, exact in every dtype; never 0 so div is defined for int.
template <typename T>
std::vector<T> seq(size_t n, int base) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<T>(base + static_cast<int>(i % 7));
    return v;
}

template <typename T>
std::vector<T> host_vec(const Tensor<T>& t) {
    std::vector<T> v(t.size());
    t.copyToHost(v.data());
    return v;
}

// Right-aligned broadcast of two row-major buffers, computed independently.
template <typename T, typename Op>
std::vector<T> reference_binary(const std::vector<T>& a, const std::vector<size_t>& as,
                                const std::vector<T>& b, const std::vector<size_t>& bs,
                                const std::vector<size_t>& os, Op op) {
    const size_t r = os.size();
    std::vector<T> out(numel(os));
    std::vector<size_t> coord(r);
    for (size_t i = 0; i < out.size(); ++i) {
        size_t tmp = i;
        for (size_t d = r; d-- > 0; ) { coord[d] = tmp % os[d]; tmp /= os[d]; }
        auto flat = [&](const std::vector<size_t>& s) {
            size_t f = 0;
            const size_t lead = r - s.size();
            for (size_t d = 0; d < s.size(); ++d)
                f = f * s[d] + (s[d] == 1 ? 0 : coord[d + lead]);
            return f;
        };
        out[i] = static_cast<T>(op(a[flat(as)], b[flat(bs)]));
    }
    return out;
}

template <typename T>
void expect_eq(const std::vector<T>& got, const std::vector<T>& want, const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_EQ(static_cast<float>(got[i]), static_cast<float>(want[i]))
            << what << " at flat index " << i;
}

struct Case { std::vector<size_t> a, b, out; };

const std::vector<Case> kCases = {
    {{32, 128},       {128},            {32, 128}},        // bias, right side
    {{128},           {32, 128},        {32, 128}},        // bias, left side
    {{4, 1},          {1, 5},           {4, 5}},           // both sides at once
    {{5},             {1},              {5}},              // size-1 vector
    {{1},             {5},              {5}},
    {{2, 3, 4},       {3, 1},           {2, 3, 4}},
    {{3, 1, 5},       {3, 4, 5},        {3, 4, 5}},        // size-1 middle axis
    {{2, 3, 4, 5},    {1, 3, 1, 5},     {2, 3, 4, 5}},     // rank 4
    {{2, 3, 1, 4, 5}, {3, 4, 1},        {2, 3, 3, 4, 5}},  // rank 5 -> _nd
    {{2, 1, 3, 2, 1, 4}, {1, 2, 3, 1, 5, 1}, {2, 2, 3, 2, 5, 4}},  // rank 6 -> _nd
    {{70000, 1, 1, 2}, {2},             {70000, 1, 1, 2}}, // gridDim.z overflow -> _nd
    {{7, 9},          {7, 9},           {7, 9}},           // no broadcast at all
};

template <typename T>
void check_case(const Case& c, const Device& dv) {
    const auto va = seq<T>(numel(c.a), 1);
    const auto vb = seq<T>(numel(c.b), 2);
    Tensor<T> a = Tensor<T>::from_vector(va, c.a).to(dv);
    Tensor<T> b = Tensor<T>::from_vector(vb, c.b).to(dv);
    const std::string tag = " " + str(c.a) + " op " + str(c.b) + " on " + dv.m_Str;

    auto run = [&](const char* name, Tensor<T> got, auto op) {
        EXPECT_EQ(got.shape(), c.out) << name << tag;
        expect_eq(host_vec(got), reference_binary<T>(va, c.a, vb, c.b, c.out, op),
                  std::string(name) + tag);
    };
    run("add", a + b, [](T x, T y) { return x + y; });
    run("sub", a - b, [](T x, T y) { return x - y; });
    run("mul", a * b, [](T x, T y) { return x * y; });
    run("div", a / b, [](T x, T y) { return div_elem(x, y); });
}

} // namespace

// ── broadcast_shapes ────────────────────────────────────────────────────────

TEST(Broadcast, ShapeRules) {
    using V = std::vector<size_t>;
    EXPECT_EQ(detail::broadcast_shapes(V{32, 128}, V{128}, "t"), (V{32, 128}));
    EXPECT_EQ(detail::broadcast_shapes(V{4, 1}, V{1, 5}, "t"), (V{4, 5}));
    EXPECT_EQ(detail::broadcast_shapes(V{2, 3, 4}, V{3, 1}, "t"), (V{2, 3, 4}));
    EXPECT_EQ(detail::broadcast_shapes(V{5}, V{1}, "t"), (V{5}));
    EXPECT_EQ(detail::broadcast_shapes(V{1}, V{5}, "t"), (V{5}));
    EXPECT_EQ(detail::broadcast_shapes(V{3, 0}, V{1}, "t"), (V{3, 0}));
}

TEST(Broadcast, IncompatibleShapesThrow) {
    using V = std::vector<size_t>;
    EXPECT_THROW(detail::broadcast_shapes(V{4}, V{5}, "t"), std::invalid_argument);
    EXPECT_THROW(detail::broadcast_shapes(V{2, 3}, V{3, 2}, "t"), std::invalid_argument);
    EXPECT_THROW(detail::broadcast_shapes(V(9, 1), V{1}, "t"), std::invalid_argument);
}

TEST(Broadcast, ExpandToStrides) {
    const auto l = detail::expand_to({128}, {1}, {32, 128});
    EXPECT_EQ(l.rank, 2u);
    EXPECT_EQ(l.stride[0], 0u);
    EXPECT_EQ(l.stride[1], 1u);
    const auto m = detail::expand_to({3, 1, 5}, {5, 5, 1}, {3, 4, 5});
    EXPECT_EQ(m.stride[0], 5u);
    EXPECT_EQ(m.stride[1], 0u);
    EXPECT_EQ(m.stride[2], 1u);
}

// ── values, every op, both backends ─────────────────────────────────────────

TEST(Broadcast, CPUFloat) {
    for (const auto& c : kCases) check_case<float>(c, Device("cpu:0"));
}

TEST(Broadcast, CPUInt) {
    for (const auto& c : kCases) check_case<int>(c, Device("cpu:0"));
}

TEST(Broadcast, GPUFloat) {
    OM_REQUIRE_CUDA();
    for (const auto& c : kCases) check_case<float>(c, Device("cuda:0"));
}

TEST(Broadcast, GPUInt) {
    OM_REQUIRE_CUDA();
    for (const auto& c : kCases) check_case<int>(c, Device("cuda:0"));
}

// float16_t division rounds differently on the device (__hdiv) than the float
// reference, so half precision checks add and mul only.
TEST(Broadcast, GPUHalfBias) {
    OM_REQUIRE_CUDA();
    Device gpu("cuda:0");
    const std::vector<size_t> xs{16, 33}, bs{33};
    const auto vx = seq<float16_t>(numel(xs), 1);
    const auto vb = seq<float16_t>(numel(bs), 2);
    Tensor<float16_t> x = Tensor<float16_t>::from_vector(vx, xs).to(gpu);
    Tensor<float16_t> b = Tensor<float16_t>::from_vector(vb, bs).to(gpu);
    auto f = [](float16_t p, float16_t q) { return static_cast<float16_t>(static_cast<float>(p) + static_cast<float>(q)); };
    auto g = [](float16_t p, float16_t q) { return static_cast<float16_t>(static_cast<float>(p) * static_cast<float>(q)); };
    expect_eq(host_vec(x + b), reference_binary<float16_t>(vx, xs, vb, bs, xs, f), "half add");
    expect_eq(host_vec(x * b), reference_binary<float16_t>(vx, xs, vb, bs, xs, g), "half mul");
}

// ── in-place and destination-provided forms ─────────────────────────────────

namespace {
void check_inplace(const Device& dv) {
    const std::vector<size_t> xs{32, 128}, bs{128};
    const auto vx = seq<float>(numel(xs), 1);
    const auto vb = seq<float>(numel(bs), 2);
    Tensor<float> x = Tensor<float>::from_vector(vx, xs).to(dv);
    Tensor<float> b = Tensor<float>::from_vector(vb, bs).to(dv);

    const void* before = x.view().data;
    x.add_(b);
    x += b;
    EXPECT_EQ(x.view().data, before) << "in-place broadcast must not reallocate";
    auto plus = [](float p, float q) { return p + q; };
    auto once = reference_binary<float>(vx, xs, vb, bs, xs, plus);
    expect_eq(host_vec(x), reference_binary<float>(once, xs, vb, bs, xs, plus), "x += b twice");

    // The result (32,128) cannot land in b's (128,) buffer.
    EXPECT_THROW(b.add_(x), std::invalid_argument);

    Tensor<float> out = Tensor<float>::zeros(xs, dv);
    const auto a = Tensor<float>::from_vector(vx, xs).to(dv);
    a.mul_out(b, out);
    expect_eq(host_vec(out), reference_binary<float>(vx, xs, vb, bs, xs,
              [](float p, float q) { return p * q; }), "mul_out");

    Tensor<float> wrong = Tensor<float>::zeros({128}, dv);
    EXPECT_THROW(a.add_out(b, wrong), std::invalid_argument);
}

void check_fused(const Device& dv) {
    const std::vector<size_t> xs{8, 3, 5}, bs{3, 1};
    const auto vx = seq<float>(numel(xs), 1);
    const auto vb = seq<float>(numel(bs), 2);
    Tensor<float> x = Tensor<float>::from_vector(vx, xs).to(dv);
    Tensor<float> b = Tensor<float>::from_vector(vb, bs).to(dv);

    Tensor<float> y = x.fused_add_mul(b, 2.0f);
    EXPECT_EQ(y.shape(), xs);
    expect_eq(host_vec(y), reference_binary<float>(vx, xs, vb, bs, xs,
              [](float p, float q) { return (p + q) * 2.0f; }), "fused_add_mul");

    // Broadcast on the left through apply_binary.
    Tensor<float> z = b.apply_binary(x, BinarySub<float>{});
    EXPECT_EQ(z.shape(), xs);
    expect_eq(host_vec(z), reference_binary<float>(vb, bs, vx, xs, xs,
              [](float p, float q) { return p - q; }), "apply_binary left");
}
} // namespace

TEST(Broadcast, CPUInPlaceAndOut) { check_inplace(Device("cpu:0")); }

TEST(Broadcast, GPUInPlaceAndOut) {
    OM_REQUIRE_CUDA();
    check_inplace(Device("cuda:0"));
}

TEST(Broadcast, CPUFused) { check_fused(Device("cpu:0")); }

TEST(Broadcast, GPUFused) {
    OM_REQUIRE_CUDA();
    check_fused(Device("cuda:0"));
}

TEST(Broadcast, IncompatibleOperandsThrow) {
    Tensor<float> a = Tensor<float>::zeros({2, 3});
    Tensor<float> b = Tensor<float>::zeros({3, 2});
    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a.apply_binary(b, BinaryAdd<float>{}), std::invalid_argument);
}
