// In-place ops (add_, mul_, relu_, fill_, …) and the destination-provided
// overloads (add_out, matmul_out, …).
//
// Two things are being asserted throughout, and the second is the point of the
// feature: that the result is right, and that producing it did not move the
// tensor's buffer. data_ptr equality before and after is what proves no
// allocation happened — a correct-but-reallocating implementation would pass
// every value check here.
#include "test_helpers.h"
#include <numeric>

namespace {

Tensor<float> host_seq(const std::vector<size_t>& shape, float start = 1.0f)
{
    size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + static_cast<float>(i);
    return Tensor<float>::from_vector(v, shape);
}

Tensor<float> dev_seq(const std::vector<size_t>& shape, float start = 1.0f)
{
    return host_seq(shape, start).cuda();
}

const void* buf(const Tensor<float>& t) { return t.view().data; }

} // namespace

// ── in-place, CPU ───────────────────────────────────────────────────────────

TEST(InPlace, CPUTensorOps) {
    Tensor<float> a = host_seq({4});            // 1 2 3 4
    Tensor<float> b = Tensor<float>::full({4}, 2.0f);
    const void* p = buf(a);

    a.add_(b);
    EXPECT_EQ(to_host(a), (std::vector<float>{3, 4, 5, 6}));
    a.sub_(b);
    EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4}));
    a.mul_(b);
    EXPECT_EQ(to_host(a), (std::vector<float>{2, 4, 6, 8}));
    a.div_(b);
    EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4}));

    EXPECT_EQ(buf(a), p) << "in-place op reallocated the buffer";
}

TEST(InPlace, CPUScalarOps) {
    Tensor<float> a = host_seq({4});
    const void* p = buf(a);

    a.add_(1.0f).mul_(2.0f).sub_(2.0f).div_(2.0f);
    EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4}));
    EXPECT_EQ(buf(a), p);
}

TEST(InPlace, CPUCompoundOperators) {
    Tensor<float> a = host_seq({4});
    Tensor<float> b = Tensor<float>::full({4}, 3.0f);
    const void* p = buf(a);

    a += b;   EXPECT_EQ(to_host(a), (std::vector<float>{4, 5, 6, 7}));
    a -= b;   EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4}));
    a *= 2.0f; EXPECT_EQ(to_host(a), (std::vector<float>{2, 4, 6, 8}));
    a /= 2.0f; EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4}));

    EXPECT_EQ(buf(a), p);
}

TEST(InPlace, CPUUnaryAndFill) {
    Tensor<float> a = Tensor<float>::from_vector({-2.0f, -1.0f, 0.0f, 3.0f}, {4});
    const void* p = buf(a);

    a.relu_();
    EXPECT_EQ(to_host(a), (std::vector<float>{0, 0, 0, 3}));

    a.fill_(7.0f);
    EXPECT_EQ(to_host(a), (std::vector<float>{7, 7, 7, 7}));

    a.apply_(Mul<float>{0.5f});
    EXPECT_EQ(to_host(a), (std::vector<float>{3.5f, 3.5f, 3.5f, 3.5f}));

    Tensor<float> b = Tensor<float>::full({4}, 2.0f);
    a.apply_binary_(b, BinaryMul<float>{});
    EXPECT_EQ(to_host(a), (std::vector<float>{7, 7, 7, 7}));

    EXPECT_EQ(buf(a), p);
}

TEST(InPlace, CPUSigmoidMatchesOutOfPlace) {
    Tensor<float> a = Tensor<float>::from_vector({-1.0f, 0.0f, 1.0f, 2.0f}, {4});
    Tensor<float> expected = a.sigmoid();
    a.sigmoid_();
    const auto got = to_host(a), want = to_host(expected);
    for (size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], want[i]);
}

// The rhs must survive the call unchanged even when it shares nothing but the
// shape — the elementwise loop writes only into the destination.
TEST(InPlace, RhsIsNotModified) {
    Tensor<float> a = host_seq({4});
    Tensor<float> b = host_seq({4}, 10.0f);
    a.add_(b);
    EXPECT_EQ(to_host(b), (std::vector<float>{10, 11, 12, 13}));
}

// x.add_(x) is the degenerate case: destination and *both* operands alias.
TEST(InPlace, SelfOperandDoubles) {
    Tensor<float> a = host_seq({4});
    a.add_(a);
    EXPECT_EQ(to_host(a), (std::vector<float>{2, 4, 6, 8}));
}

// ── in-place, GPU ───────────────────────────────────────────────────────────

TEST(InPlace, GPUTensorOps) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = dev_seq({4});
    Tensor<float> b = Tensor<float>::full({4}, 2.0f).cuda();
    const void* p = buf(a);

    a.add_(b);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{3, 4, 5, 6}));
    a.mul_(b);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{6, 8, 10, 12}));
    a.sub_(b);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{4, 6, 8, 10}));
    a.div_(b);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{2, 3, 4, 5}));

    EXPECT_EQ(buf(a), p);
}

TEST(InPlace, GPUScalarAndUnary) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = Tensor<float>::from_vector({-2.0f, -1.0f, 1.0f, 2.0f}, {4}).cuda();
    const void* p = buf(a);

    a.mul_(2.0f);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{-4, -2, 2, 4}));
    a.relu_();
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{0, 0, 2, 4}));
    a.fill_(1.5f);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{1.5f, 1.5f, 1.5f, 1.5f}));
    a.apply_(Add<float>{0.5f});
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{2, 2, 2, 2}));

    EXPECT_EQ(buf(a), p);
}

TEST(InPlace, GPUSelfOperandDoubles) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = dev_seq({4});
    a.add_(a);
    EXPECT_EQ(to_host(a.cpu()), (std::vector<float>{2, 4, 6, 8}));
}

// The packed path (pack_width > 1) and its sub-pack tail are a separate code
// path from the scalar one; run a size that leaves a tail through it aliased.
TEST(InPlace, GPUPackedTailAliased) {
    OM_REQUIRE_CUDA();
    for (size_t n : {size_t{1}, size_t{7}, size_t{1021}, size_t{4096}}) {
        std::vector<char> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>(i % 5);
        Tensor<char> a = Tensor<char>::from_vector(v, {n}).cuda();
        a.add_(a);
        std::vector<char> got(n);
        a.cpu().copyToHost(got.data());
        for (size_t i = 0; i < n; ++i)
            ASSERT_EQ(got[i], static_cast<char>(2 * (i % 5))) << "n=" << n << " i=" << i;
    }
}

// Every rank goes through the same contiguous fast path, but the fallback is
// selected per launcher; check the ranks the launchers special-case.
TEST(InPlace, GPUEveryRank) {
    OM_REQUIRE_CUDA();
    const std::vector<std::vector<size_t>> shapes = {
        {64}, {8, 8}, {4, 4, 4}, {2, 4, 2, 4}, {2, 2, 2, 2, 4}
    };
    for (const auto& sh : shapes) {
        Tensor<float> a = dev_seq(sh);
        Tensor<float> want = a.add(a);
        a.add_(a);
        EXPECT_EQ(to_host(a.cpu()), to_host(want.cpu())) << "rank " << sh.size();
    }
}

// ── in-place on a stream ────────────────────────────────────────────────────

TEST(InPlace, GPUStreamOverload) {
    OM_REQUIRE_CUDA();
    Stream s;
    Tensor<float> a = dev_seq({256});
    Tensor<float> b = Tensor<float>::full({256}, 2.0f).cuda();
    const void* p = buf(a);

    a.mul_(b, s).add_(1.0f, s).relu_(s);
    s.synchronize();

    const auto got = to_host(a.cpu());
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_FLOAT_EQ(got[i], 2.0f * static_cast<float>(i + 1) + 1.0f);
    EXPECT_EQ(buf(a), p);
}

// ── destination-provided overloads ──────────────────────────────────────────

TEST(OutParam, CPUReusesOneDestination) {
    Tensor<float> a = host_seq({4});
    Tensor<float> b = Tensor<float>::full({4}, 2.0f);
    Tensor<float> out({4});
    const void* p = buf(out);

    a.add_out(b, out);
    EXPECT_EQ(to_host(out), (std::vector<float>{3, 4, 5, 6}));
    a.mul_out(b, out);
    EXPECT_EQ(to_host(out), (std::vector<float>{2, 4, 6, 8}));
    a.mul_out(3.0f, out);
    EXPECT_EQ(to_host(out), (std::vector<float>{3, 6, 9, 12}));
    a.relu_out(out);
    EXPECT_EQ(to_host(out), (std::vector<float>{1, 2, 3, 4}));
    a.apply_binary_out(b, BinaryAdd<float>{}, out);
    EXPECT_EQ(to_host(out), (std::vector<float>{3, 4, 5, 6}));

    EXPECT_EQ(buf(out), p) << "the destination was replaced instead of written";
    EXPECT_EQ(to_host(a), (std::vector<float>{1, 2, 3, 4})) << "operand was clobbered";
}

TEST(OutParam, ReturnsTheDestination) {
    Tensor<float> a = host_seq({4});
    Tensor<float> out({4});
    Tensor<float>& r = a.add_out(1.0f, out);
    EXPECT_EQ(&r, &out);
}

TEST(OutParam, GPUReusesOneDestination) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = dev_seq({4});
    Tensor<float> b = Tensor<float>::full({4}, 2.0f).cuda();
    Tensor<float> out({4}, Device("cuda:0"));
    const void* p = buf(out);

    a.add_out(b, out);
    EXPECT_EQ(to_host(out.cpu()), (std::vector<float>{3, 4, 5, 6}));
    a.div_out(b, out);
    EXPECT_EQ(to_host(out.cpu()), (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
    EXPECT_EQ(buf(out), p);
}

TEST(OutParam, MatmulTransposePermute) {
    Tensor<float> a = host_seq({2, 3});          // [[1,2,3],[4,5,6]]
    Tensor<float> b = Tensor<float>::full({3, 2}, 1.0f);

    Tensor<float> mm({2, 2});
    a.matmul_out(b, mm);
    EXPECT_EQ(to_host(mm), to_host(a.matmul(b)));

    Tensor<float> tr({3, 2});
    a.transpose_out(tr);
    EXPECT_EQ(to_host(tr), to_host(a.transpose()));

    Tensor<float> pm({3, 2});
    a.permute_out({1, 0}, pm);
    EXPECT_EQ(to_host(pm), to_host(a.permute({1, 0})));
}

TEST(OutParam, GPUMatmulTransposePermute) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = dev_seq({2, 3});
    Tensor<float> b = Tensor<float>::full({3, 2}, 1.0f).cuda();
    Device gpu("cuda:0");

    Tensor<float> mm({2, 2}, gpu);
    a.matmul_out(b, mm);
    EXPECT_EQ(to_host(mm.cpu()), to_host(a.matmul(b).cpu()));

    Tensor<float> tr({3, 2}, gpu);
    a.transpose_out(tr);
    EXPECT_EQ(to_host(tr.cpu()), to_host(a.transpose().cpu()));

    Tensor<float> pm({3, 2}, gpu);
    a.permute_out({1, 0}, pm);
    EXPECT_EQ(to_host(pm.cpu()), to_host(a.permute({1, 0}).cpu()));
}

// ── validation ──────────────────────────────────────────────────────────────

TEST(OutParam, WrongDestinationShapeThrows) {
    Tensor<float> a = host_seq({4});
    Tensor<float> out({5});
    EXPECT_THROW(a.add_out(1.0f, out), std::invalid_argument);
    EXPECT_THROW(a.relu_out(out), std::invalid_argument);
}

TEST(OutParam, WrongDestinationDeviceThrows) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = dev_seq({4});
    Tensor<float> out({4});                       // host destination, device operand
    EXPECT_THROW(a.add_out(1.0f, out), std::invalid_argument);
}

TEST(OutParam, MismatchedOperandThrows) {
    Tensor<float> a = host_seq({4});
    Tensor<float> b = host_seq({5});
    Tensor<float> out({4});
    EXPECT_THROW(a.add_out(b, out), std::invalid_argument);
    EXPECT_THROW(a.add_(b), std::invalid_argument);
}

TEST(OutParam, MixedDeviceOperandsThrow) {
    OM_REQUIRE_CUDA();
    Tensor<float> host = host_seq({4});
    Tensor<float> dev = dev_seq({4});
    EXPECT_THROW(host.add_(dev), std::invalid_argument);
    EXPECT_THROW(dev.add_(host), std::invalid_argument);
}

// matmul, transpose and permute read elements they do not write, so unlike the
// elementwise family they cannot service an aliased destination — and say so
// instead of returning a plausible wrong answer.
TEST(OutParam, AliasedDestinationRejectedForNonElementwise) {
    Tensor<float> a = host_seq({2, 2});
    Tensor<float> b = Tensor<float>::full({2, 2}, 1.0f);
    EXPECT_THROW(a.matmul_out(b, a), std::invalid_argument);
    EXPECT_THROW(a.matmul_out(b, b), std::invalid_argument);
    EXPECT_THROW(a.transpose_out(a), std::invalid_argument);
    EXPECT_THROW(a.permute_out({1, 0}, a), std::invalid_argument);
}

// ── the reason the feature exists ───────────────────────────────────────────

// A loop of out-of-place ops allocates a fresh result every iteration; the
// same loop written in-place keeps one buffer for its whole life.
TEST(InPlace, LoopKeepsOneBuffer) {
    Tensor<float> w = Tensor<float>::zeros({1024});
    Tensor<float> g = Tensor<float>::full({1024}, 0.5f);
    const void* p = buf(w);

    for (int step = 0; step < 100; ++step) {
        w.add_(g);
        ASSERT_EQ(buf(w), p) << "buffer moved at step " << step;
    }
    const auto got = to_host(w);
    EXPECT_FLOAT_EQ(got.front(), 50.0f);
    EXPECT_FLOAT_EQ(got.back(), 50.0f);
}

TEST(OutParam, LoopKeepsOneDestination) {
    OM_REQUIRE_CUDA();
    Tensor<float> a = Tensor<float>::full({1024}, 1.0f).cuda();
    Tensor<float> b = Tensor<float>::full({1024}, 2.0f).cuda();
    Tensor<float> out({1024}, Device("cuda:0"));
    const void* p = buf(out);

    for (int step = 0; step < 100; ++step) {
        a.mul_out(b, out);
        ASSERT_EQ(buf(out), p) << "destination moved at step " << step;
    }
    EXPECT_FLOAT_EQ(to_host(out.cpu()).front(), 2.0f);
}
