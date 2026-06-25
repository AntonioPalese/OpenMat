#include "test_helpers.h"
#include "stream.h"

// ── om::Stream construction ───────────────────────────────────────────────────

TEST(Stream, ConstructAndDestroy) {
    EXPECT_NO_THROW({ om::Stream s; });
}

TEST(Stream, GetReturnsNonNull) {
    om::Stream s;
    EXPECT_NE(s.get(), nullptr);
}

TEST(Stream, DefaultStreamIsNull) {
    auto s = om::Stream::default_stream();
    EXPECT_EQ(s.get(), nullptr);
}

TEST(Stream, MoveConstruct) {
    om::Stream s1;
    cudaStream_t raw = s1.get();
    om::Stream s2(std::move(s1));
    EXPECT_EQ(s2.get(), raw);
    EXPECT_EQ(s1.get(), nullptr);
}

TEST(Stream, Synchronize) {
    om::Stream s;
    EXPECT_NO_THROW(s.synchronize());
}

// ── Stream overloads produce correct results ─────────────────────────────────

TEST(StreamOps, AddBinaryMatchesDefault) {
    auto a = Tensor<float>::from_vector({1,2,3,4}, {2,2}, Device("cuda:0"));
    auto b = Tensor<float>::from_vector({10,20,30,40}, {2,2}, Device("cuda:0"));

    auto ref = a + b;

    om::Stream s;
    auto res = a.add(b, s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    ASSERT_EQ(vref.size(), vres.size());
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, SubBinaryMatchesDefault) {
    auto a = Tensor<float>::from_vector({10,20,30,40}, {4}, Device("cuda:0"));
    auto b = Tensor<float>::from_vector({1,2,3,4},     {4}, Device("cuda:0"));

    auto ref = a - b;

    om::Stream s;
    auto res = a.sub(b, s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, MulScalarMatchesDefault) {
    auto a = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3}, Device("cuda:0"));

    auto ref = a * 3.0f;

    om::Stream s;
    auto res = a.mul(3.0f, s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, MatmulMatchesDefault) {
    auto a = Tensor<float>::from_vector({1,2,3,4}, {2,2}, Device("cuda:0"));
    auto b = Tensor<float>::from_vector({5,6,7,8}, {2,2}, Device("cuda:0"));

    auto ref = a.matmul(b);

    om::Stream s;
    auto res = a.matmul(b, s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, TransposeMatchesDefault) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3}, Device("cuda:0"));

    auto ref = t.transpose();

    om::Stream s;
    auto res = t.transpose(s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, PermuteMatchesDefault) {
    auto t = Tensor<float>::ones({2,3,4}, Device("cuda:0"));

    auto ref = t.permute({2,0,1});

    om::Stream s;
    auto res = t.permute({2,0,1}, s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, ReLUMatchesDefault) {
    auto t = Tensor<float>::from_vector({-2,-1,0,1,2,3}, {2,3}, Device("cuda:0"));

    auto ref = t.relu();

    om::Stream s;
    auto res = t.relu(s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

TEST(StreamOps, SigmoidMatchesDefault) {
    auto t = Tensor<float>::from_vector({-1,0,1,2}, {4}, Device("cuda:0"));

    auto ref = t.sigmoid();

    om::Stream s;
    auto res = t.sigmoid(s);
    s.synchronize();

    auto vref = to_host(ref);
    auto vres = to_host(res);
    for (size_t i = 0; i < vref.size(); ++i)
        EXPECT_FLOAT_EQ(vref[i], vres[i]);
}

// ── Two independent streams produce correct results ───────────────────────────

TEST(StreamOps, TwoIndependentStreams) {
    auto a = Tensor<float>::from_vector({1,2,3,4}, {4}, Device("cuda:0"));
    auto b = Tensor<float>::from_vector({10,20,30,40}, {4}, Device("cuda:0"));
    auto c = Tensor<float>::from_vector({5,5,5,5}, {4}, Device("cuda:0"));

    om::Stream s1, s2;

    // Both kernels enqueued asynchronously on separate streams
    auto r1 = a.add(b, s1);   // [11,22,33,44]
    auto r2 = a.mul(c, s2);   // [5,10,15,20]

    s1.synchronize();
    s2.synchronize();

    auto v1 = to_host(r1);
    auto v2 = to_host(r2);

    EXPECT_FLOAT_EQ(v1[0], 11.f);
    EXPECT_FLOAT_EQ(v1[3], 44.f);
    EXPECT_FLOAT_EQ(v2[0], 5.f);
    EXPECT_FLOAT_EQ(v2[3], 20.f);
}

// ── default_stream() behaves synchronously (backward compat) ─────────────────

TEST(StreamOps, DefaultStreamSyncBehavior) {
    auto a = Tensor<float>::from_vector({1,2,3,4}, {4}, Device("cuda:0"));
    auto b = Tensor<float>::from_vector({1,1,1,1}, {4}, Device("cuda:0"));

    // Using default stream = synchronous, result immediately ready
    auto s = om::Stream::default_stream();
    auto r = a.add(b, s);
    // no explicit synchronize needed — default stream is synchronous

    auto v = to_host(r);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i + 2));
}
