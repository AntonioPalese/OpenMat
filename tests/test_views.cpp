// Views: reshape/squeeze/unsqueeze/slice/select sharing a Storage, and every
// op reading or writing a non-contiguous view.
//
// Each check runs on both backends. The GPU kernels have three paths per op
// (contiguous fast path, rank-specialized, _nd) and only the last two see a
// strided view, so the shapes here are picked to reach both: rank 2/3 views
// for the rank-specialized kernels, a rank-5 view for _nd.
#include "test_helpers.h"
#include <functional>
#include <numeric>

namespace {

const Device kCPU(0, DEVICE_TYPE::CPU);
const Device kCUDA(0, DEVICE_TYPE::CUDA);

// 1, 2, 3, … laid out row-major in `shape`, on `dv`.
Tensor<float> seq(const std::vector<size_t>& shape, const Device& dv, float start = 1.0f)
{
    size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + static_cast<float>(i);
    return Tensor<float>::from_vector(v, shape, dv);
}

// The value seq() put at a row-major flat index.
float seq_at(size_t flat, float start = 1.0f) { return start + static_cast<float>(flat); }

// Visits every multi-index of `shape` in row-major order.
void for_each_index(const std::vector<size_t>& shape,
                    const std::function<void(const std::vector<size_t>&)>& fn)
{
    size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    std::vector<size_t> idx(shape.size());
    for (size_t flat = 0; flat < n; ++flat) {
        size_t tmp = flat;
        for (size_t d = shape.size(); d-- > 0; ) { idx[d] = tmp % shape[d]; tmp /= shape[d]; }
        fn(idx);
    }
}

size_t row_major(const std::vector<size_t>& idx, const std::vector<size_t>& shape)
{
    size_t flat = 0;
    for (size_t d = 0; d < shape.size(); ++d) flat = flat * shape[d] + idx[d];
    return flat;
}

// Runs `body` on the CPU, then on the GPU when one is present. Defined as a
// macro pair so a GPU-less machine reports the CUDA half as skipped rather
// than silently passing it.
#define VIEW_TEST(name)                                          \
    void name##_body(const Device& dv);                          \
    TEST(Views, name##_CPU) { name##_body(kCPU); }               \
    TEST(Views, name##_CUDA) { OM_REQUIRE_CUDA(); name##_body(kCUDA); } \
    void name##_body(const Device& dv)

} // namespace

// ── construction ────────────────────────────────────────────────────────────

VIEW_TEST(ReshapeOfContiguousIsAView)
{
    auto a = seq({2, 3}, dv);
    auto r = a.reshape({3, 2});
    EXPECT_TRUE(r.shares_storage(a));
    EXPECT_EQ(r.view().data, a.view().data);
    r.fill_(7.0f);
    EXPECT_EQ(to_host(a), std::vector<float>(6, 7.0f));

    auto f = a.flatten();
    EXPECT_TRUE(f.shares_storage(a));
    EXPECT_EQ(f.shape(), (std::vector<size_t>{6}));
}

VIEW_TEST(ReshapeOfStridedViewCopies)
{
    auto a = seq({2, 4}, dv);                 // 1..8
    auto s = a.slice(1, 0, 4, 2);             // columns 0 and 2: 1 3 / 5 7
    ASSERT_FALSE(s.is_contiguous());
    auto r = s.reshape({4});
    EXPECT_FALSE(r.shares_storage(a));
    EXPECT_TRUE(r.is_contiguous());
    EXPECT_EQ(to_host(r), (std::vector<float>{1, 3, 5, 7}));
}

VIEW_TEST(SqueezeUnsqueezeAreViews)
{
    auto a = seq({2, 1, 3}, dv);
    auto s = a.squeeze(1);
    EXPECT_TRUE(s.shares_storage(a));
    EXPECT_EQ(s.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(s.stride(), (std::vector<size_t>{3, 1}));

    auto u = s.unsqueeze(0);
    EXPECT_TRUE(u.shares_storage(a));
    EXPECT_EQ(u.shape(), (std::vector<size_t>{1, 2, 3}));
    EXPECT_TRUE(u.is_contiguous());

    // Unsqueeze of a strided view keeps it strided and still aliasing.
    auto col = seq({3, 4}, dv).select(1, 2);  // 3 7 11, stride 4
    auto cu = col.unsqueeze(1);
    EXPECT_EQ(cu.shape(), (std::vector<size_t>{3, 1}));
    EXPECT_EQ(to_host(cu), (std::vector<float>{3, 7, 11}));
}

VIEW_TEST(SliceAndSelectValues)
{
    auto a = seq({3, 4}, dv);                 // 1..12

    auto cols = a.slice(1, 1, 4, 2);          // columns 1 and 3
    EXPECT_EQ(cols.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(cols.stride(), (std::vector<size_t>{4, 2}));
    EXPECT_EQ(cols.storage_offset(), 1u);
    EXPECT_FALSE(cols.is_contiguous());
    EXPECT_EQ(to_host(cols), (std::vector<float>{2, 4, 6, 8, 10, 12}));

    auto row = a.select(0, 1);
    EXPECT_TRUE(row.is_contiguous());
    EXPECT_EQ(row.storage_offset(), 4u);
    EXPECT_EQ(to_host(row), (std::vector<float>{5, 6, 7, 8}));

    auto col = a.select(1, 2);
    EXPECT_FALSE(col.is_contiguous());
    EXPECT_EQ(to_host(col), (std::vector<float>{3, 7, 11}));

    // Chained: rows 1..2 of the column slice, then one element.
    auto inner = cols.slice(0, 1, 3);
    EXPECT_EQ(to_host(inner), (std::vector<float>{6, 8, 10, 12}));
    EXPECT_EQ(to_host(inner.select(0, 1).select(0, 0)), (std::vector<float>{10}));

    // Out-of-range stop clamps like a Python slice; an empty slice is legal.
    EXPECT_EQ(a.slice(0, 1, 99).shape(), (std::vector<size_t>{2, 4}));
    auto empty = a.slice(0, 3, 3);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ((empty + empty).size(), 0u);

    EXPECT_THROW(a.slice(2, 0, 1), std::out_of_range);
    EXPECT_THROW(a.slice(0, 0, 3, 0), std::invalid_argument);
    EXPECT_THROW(a.select(0, 3), std::out_of_range);
}

// ── ops on strided views ────────────────────────────────────────────────────

// Two interleaved column views of one (R, 2C) buffer — both strided, the
// shape the rank-specialized kernels see — combined every way an op can take
// them, and checked element by element against the same formula on the host.
VIEW_TEST(ElementwiseOnStridedViews)
{
    const size_t R = 5, C = 7;
    auto x = seq({R, 2 * C}, dv);
    auto even = x.slice(1, 0, 2 * C, 2);      // x[:, 0::2]
    auto odd  = x.slice(1, 1, 2 * C, 2);      // x[:, 1::2]
    ASSERT_FALSE(even.is_contiguous());

    auto expect = [&](const Tensor<float>& t, const std::function<float(float, float)>& f) {
        auto h = to_host(t);
        ASSERT_EQ(h.size(), R * C);
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c) {
                float e = seq_at(r * 2 * C + 2 * c), o = seq_at(r * 2 * C + 2 * c + 1);
                EXPECT_FLOAT_EQ(h[r * C + c], f(e, o)) << "at (" << r << ", " << c << ")";
            }
    };

    expect(even + odd,            [](float e, float o) { return e + o; });
    expect(odd - even,            [](float e, float o) { return o - e; });
    expect(even * odd,            [](float e, float o) { return e * o; });
    expect(odd / even,            [](float e, float o) { return o / e; });
    expect(even * 3.0f,           [](float e, float)   { return e * 3.0f; });
    expect(odd - 20.0f,           [](float, float o)   { return o - 20.0f; });
    expect((odd - 20.0f).relu(),  [](float, float o)   { return o - 20.0f > 0 ? o - 20.0f : 0.0f; });
    expect(even.relu(),           [](float e, float)   { return e; });
    expect(even.scale_shift(2.0f, 1.0f), [](float e, float) { return e * 2.0f + 1.0f; });
    expect(even.fused_add_mul(odd, 0.5f), [](float e, float o) { return (e + o) * 0.5f; });

    // A strided operand broadcast against a contiguous one.
    auto bias = seq({C}, dv, 100.0f);
    auto h = to_host(even + bias);
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c)
            EXPECT_FLOAT_EQ(h[r * C + c], seq_at(r * 2 * C + 2 * c) + seq_at(c, 100.0f));
}

// Rank 5 goes through the _nd kernels, which had shared one offset between
// source and destination — only correct while every stride matched.
VIEW_TEST(RankFiveViewsTakeTheNdPath)
{
    const std::vector<size_t> full{2, 2, 3, 2, 6};
    auto x = seq(full, dv);
    auto v = x.slice(4, 1, 6, 2);             // last axis 1, 3, 5
    const std::vector<size_t> vshape{2, 2, 3, 2, 3};
    ASSERT_EQ(v.shape(), vshape);

    auto check = [&](const Tensor<float>& t, const std::function<float(float)>& f) {
        auto h = to_host(t);
        for_each_index(vshape, [&](const std::vector<size_t>& i) {
            std::vector<size_t> src = i;
            src[4] = 1 + 2 * i[4];
            EXPECT_FLOAT_EQ(h[row_major(i, vshape)], f(seq_at(row_major(src, full))));
        });
    };
    check(v + v,        [](float a) { return a + a; });
    check(v * 2.0f,     [](float a) { return a * 2.0f; });
    check(v.sigmoid(),  [](float a) { return 1.0f / (1.0f + std::exp(-a)); });
    check(v.contiguous(), [](float a) { return a; });
}

VIEW_TEST(WritesThroughViews)
{
    auto x = Tensor<float>::zeros({3, 4}, dv);
    x.slice(1, 1, 3).add_(1.0f);              // columns 1..2 += 1
    x.select(0, 2).fill_(5.0f);               // last row = 5
    EXPECT_EQ(to_host(x), (std::vector<float>{0, 1, 1, 0,
                                              0, 1, 1, 0,
                                              5, 5, 5, 5}));

    // copy_ broadcasts a row into a strided view.
    auto y = Tensor<float>::zeros({3, 4}, dv);
    auto row = Tensor<float>::from_vector({1, 2}, {2}, dv);
    y.slice(1, 0, 4, 2).copy_(row);           // columns 0 and 2 ← 1, 2
    EXPECT_EQ(to_host(y), (std::vector<float>{1, 0, 2, 0,
                                              1, 0, 2, 0,
                                              1, 0, 2, 0}));

    // _out into a strided destination writes only its own elements.
    auto z = Tensor<float>::zeros({2, 4}, dv);
    auto dst = z.slice(1, 1, 4, 2);           // columns 1 and 3
    auto a = Tensor<float>::from_vector({1, 2, 3, 4}, {2, 2}, dv);
    a.add_out(a, dst);
    EXPECT_EQ(to_host(z), (std::vector<float>{0, 2, 0, 4,
                                              0, 6, 0, 8}));
    a.relu_out(dst);
    EXPECT_EQ(to_host(z), (std::vector<float>{0, 1, 0, 2,
                                              0, 3, 0, 4}));
}

VIEW_TEST(ReductionsAndMatmulOnViews)
{
    auto x = seq({4, 6}, dv);
    auto cols = x.slice(1, 0, 6, 2);          // columns 0, 2, 4
    float s = 0, mn = 1e9f, mx = -1e9f;
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 6; c += 2) {
            float v = seq_at(r * 6 + c);
            s += v; mn = std::min(mn, v); mx = std::max(mx, v);
        }
    EXPECT_FLOAT_EQ(cols.sum(), s);
    EXPECT_FLOAT_EQ(cols.min(), mn);
    EXPECT_FLOAT_EQ(cols.max(), mx);
    EXPECT_FLOAT_EQ(cols.mean(), s / 12.0f);

    auto b = seq({3, 2}, dv);                 // 1 2 / 3 4 / 5 6
    auto m = cols.matmul(b);                  // (4,3) @ (3,2)
    auto h = to_host(m);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j) {
            float e = 0;
            for (size_t k = 0; k < 3; ++k) e += seq_at(i * 6 + 2 * k) * seq_at(k * 2 + j);
            EXPECT_FLOAT_EQ(h[i * 2 + j], e);
        }

    // matmul/transpose/permute write their destination as one flat run.
    auto big = Tensor<float>::zeros({4, 4}, dv);
    auto strided_out = big.slice(1, 0, 4, 2);
    EXPECT_THROW(cols.matmul_out(b, strided_out), std::invalid_argument);
}

VIEW_TEST(CopiesOfViewsAreContiguousAndIndependent)
{
    auto x = seq({3, 4}, dv);
    auto col = x.select(1, 1);                // 2 6 10
    Tensor<float> c(col);
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FALSE(c.shares_storage(x));
    x.fill_(0.0f);
    EXPECT_EQ(to_host(c), (std::vector<float>{2, 6, 10}));
    EXPECT_EQ(to_host(col), (std::vector<float>{0, 0, 0}));

    auto cl = x.slice(0, 0, 2).clone();
    EXPECT_FALSE(cl.shares_storage(x));
    EXPECT_TRUE(x.contiguous().shares_storage(x));
}

TEST(Views, TransfersOfViews)
{
    OM_REQUIRE_CUDA();
    auto h = seq({3, 4}, kCPU);
    auto d = h.slice(1, 1, 4, 2).cuda();      // columns 1 and 3
    EXPECT_TRUE(d.is_contiguous());
    EXPECT_EQ(to_host(d), (std::vector<float>{2, 4, 6, 8, 10, 12}));

    auto back = seq({3, 4}, kCUDA).select(1, 0).cpu();
    EXPECT_EQ(to_host(back), (std::vector<float>{1, 5, 9}));
}

// ── aliasing rules ──────────────────────────────────────────────────────────

VIEW_TEST(AliasingRules)
{
    auto x = seq({8}, dv);

    // The same view as destination: the in-place case, always legal.
    x.add_(x);
    EXPECT_EQ(to_host(x), (std::vector<float>{2, 4, 6, 8, 10, 12, 14, 16}));

    // Disjoint slices of one storage: legal, and correct.
    auto lo = x.slice(0, 0, 4), hi = x.slice(0, 4, 8);
    lo.add_(hi);
    EXPECT_EQ(to_host(x), (std::vector<float>{12, 16, 20, 24, 10, 12, 14, 16}));

    // Shifted overlap: the kernel would read elements it already wrote.
    auto a = x.slice(0, 0, 7), b = x.slice(0, 1, 8);
    EXPECT_THROW(a.add_(b), std::invalid_argument);
    EXPECT_THROW(b.copy_(a), std::invalid_argument);

    // A broadcast operand that is also the destination's storage.
    auto m = seq({2, 4}, dv);
    auto first_row = m.select(0, 0);
    EXPECT_THROW(m.add_(first_row), std::invalid_argument);

    // Copying a view onto itself is a no-op, not an error.
    x.copy_(x);

    // Ops that read what they do not write refuse any overlap.
    auto buf = seq({8}, dv);
    auto src = buf.slice(0, 0, 4).reshape({2, 2});   // contiguous views,
    auto dst = buf.slice(0, 2, 6).reshape({2, 2});   // overlapping by two
    EXPECT_THROW(src.transpose_out(dst), std::invalid_argument);
    auto sq = seq({4, 4}, dv);
    EXPECT_THROW(sq.matmul_out(sq, sq), std::invalid_argument);
    auto other = seq({2, 2}, dv);
    EXPECT_NO_THROW(src.transpose_out(other));       // separate storage
}

// ── lifetime ────────────────────────────────────────────────────────────────

VIEW_TEST(ViewOutlivesItsBase)
{
    Tensor<float> v = [&] {
        auto base = seq({4, 4}, dv);
        return base.select(0, 2);             // base dies here
    }();
    EXPECT_EQ(to_host(v), (std::vector<float>{9, 10, 11, 12}));
    v.add_(1.0f);
    EXPECT_EQ(to_host(v), (std::vector<float>{10, 11, 12, 13}));
}

// A view of a result allocated on a non-default stream frees on *that* stream
// when the last view dies — the stream-ownership invariant, now carried by
// the Storage. memcheck in CI is what would flag a free on the wrong stream.
TEST(Views, ViewKeepsStreamOrderedStorageAlive)
{
    OM_REQUIRE_CUDA();
    Stream s;
    auto a = seq({64, 64}, kCUDA), b = seq({64, 64}, kCUDA);
    Tensor<float> v = [&] {
        Tensor<float> r = a.add(b, s);        // allocated on s
        return r.slice(0, 10, 12);
    }();
    EXPECT_EQ(v.stream().get(), s.get());
    s.synchronize();
    auto h = to_host(v);
    ASSERT_EQ(h.size(), 128u);
    EXPECT_FLOAT_EQ(h[0], 2 * seq_at(10 * 64));
    EXPECT_FLOAT_EQ(h[127], 2 * seq_at(11 * 64 + 63));
}
