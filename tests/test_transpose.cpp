#include "test_helpers.h"

// ── transpose (rank-2 only) ───────────────────────────────────────────────────

TEST(Transpose, CPU_2x3) {
    // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    auto r = t.transpose();
    ASSERT_EQ(r.rank(), 2u);
    ASSERT_EQ(r.shape()[0], 3u);
    ASSERT_EQ(r.shape()[1], 2u);
    auto v = to_host(r);
    EXPECT_FLOAT_EQ(v[0], 1.f); // r(0,0)
    EXPECT_FLOAT_EQ(v[1], 4.f); // r(0,1)
    EXPECT_FLOAT_EQ(v[2], 2.f); // r(1,0)
    EXPECT_FLOAT_EQ(v[3], 5.f); // r(1,1)
    EXPECT_FLOAT_EQ(v[4], 3.f); // r(2,0)
    EXPECT_FLOAT_EQ(v[5], 6.f); // r(2,1)
}

TEST(Transpose, CPU_SquareMatrix) {
    auto t = Tensor<float>::from_vector({1,2,3,4}, {2,2});
    auto r = t.transpose();
    auto v = to_host(r);
    EXPECT_FLOAT_EQ(v[0], 1.f);
    EXPECT_FLOAT_EQ(v[1], 3.f);
    EXPECT_FLOAT_EQ(v[2], 2.f);
    EXPECT_FLOAT_EQ(v[3], 4.f);
}

TEST(Transpose, CPU_SumPreserved) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    EXPECT_FLOAT_EQ(t.transpose().sum(), t.sum());
}

TEST(Transpose, CPU_Roundtrip) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    auto rt = t.transpose().transpose();
    auto v_orig = to_host(t);
    auto v_rt   = to_host(rt);
    ASSERT_EQ(v_orig.size(), v_rt.size());
    for (size_t i = 0; i < v_orig.size(); ++i)
        EXPECT_FLOAT_EQ(v_orig[i], v_rt[i]);
}

TEST(Transpose, CPU_Rank1Throws) {
    auto t = Tensor<float>::from_vector({1,2,3}, {3});
    EXPECT_THROW(t.transpose(), std::runtime_error);
}

TEST(Transpose, GPU_2x3) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3}, Device("cuda:0"));
    auto r = t.transpose();
    ASSERT_EQ(r.shape()[0], 3u);
    ASSERT_EQ(r.shape()[1], 2u);
    auto v = to_host(r);
    EXPECT_FLOAT_EQ(v[0], 1.f);
    EXPECT_FLOAT_EQ(v[1], 4.f);
    EXPECT_FLOAT_EQ(v[2], 2.f);
    EXPECT_FLOAT_EQ(v[3], 5.f);
}

TEST(Transpose, GPU_SumPreserved) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3}, Device("cuda:0"));
    EXPECT_FLOAT_EQ(t.transpose().sum(), t.sum());
}

// ── permute ───────────────────────────────────────────────────────────────────

TEST(Permute, CPU_Rank2_Swap) {
    // permute({1,0}) on a 2D tensor is equivalent to transpose
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    auto p = t.permute({1, 0});
    auto r = t.transpose();
    auto vp = to_host(p);
    auto vr = to_host(r);
    ASSERT_EQ(vp.size(), vr.size());
    for (size_t i = 0; i < vp.size(); ++i)
        EXPECT_FLOAT_EQ(vp[i], vr[i]);
}

TEST(Permute, CPU_Rank3_012_Identity) {
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i);
    auto t = Tensor<float>::from_vector(data, {2, 3, 4});
    auto p = t.permute({0, 1, 2}); // identity
    auto vp = to_host(p);
    for (int i = 0; i < 24; ++i) EXPECT_FLOAT_EQ(vp[i], static_cast<float>(i));
}

TEST(Permute, CPU_Rank3_Shape) {
    auto t = Tensor<float>::ones({2, 3, 4});
    auto p = t.permute({2, 0, 1}); // (4,2,3)
    ASSERT_EQ(p.shape()[0], 4u);
    ASSERT_EQ(p.shape()[1], 2u);
    ASSERT_EQ(p.shape()[2], 3u);
    EXPECT_FLOAT_EQ(p.sum(), 24.f);
}

TEST(Permute, CPU_Rank3_Values) {
    // t[i,j,k] = i*12 + j*4 + k  (row-major, shape 2x3x4)
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i);
    auto t = Tensor<float>::from_vector(data, {2, 3, 4});

    // axes = {1,2,0}: out[j,k,i] = t[i,j,k]
    auto p = t.permute({1, 2, 0});
    ASSERT_EQ(p.shape()[0], 3u);
    ASSERT_EQ(p.shape()[1], 4u);
    ASSERT_EQ(p.shape()[2], 2u);

    // p(0,0,0) = t(0,0,0) = 0
    // p(1,2,1) = t(1,1,2) = 1*12 + 1*4 + 2 = 18
    auto v = to_host(p);
    // flat index of p(1,2,1) in row-major {3,4,2}: 1*8 + 2*2 + 1 = 13
    EXPECT_FLOAT_EQ(v[0],  0.f);
    EXPECT_FLOAT_EQ(v[13], 18.f);
}

TEST(Permute, CPU_DuplicateAxisThrows) {
    auto t = Tensor<float>::ones({2, 3});
    EXPECT_THROW(t.permute({0, 0}), std::invalid_argument);
}

TEST(Permute, CPU_WrongLengthThrows) {
    auto t = Tensor<float>::ones({2, 3});
    EXPECT_THROW(t.permute({0, 1, 2}), std::invalid_argument);
}

TEST(Permute, CPU_OutOfRangeAxisThrows) {
    auto t = Tensor<float>::ones({2, 3});
    EXPECT_THROW(t.permute({0, 5}), std::out_of_range);
}

TEST(Permute, GPU_Rank3_Shape) {
    auto t = Tensor<float>::ones({2, 3, 4}, Device("cuda:0"));
    auto p = t.permute({2, 0, 1});
    ASSERT_EQ(p.shape()[0], 4u);
    ASSERT_EQ(p.shape()[1], 2u);
    ASSERT_EQ(p.shape()[2], 3u);
    EXPECT_FLOAT_EQ(p.sum(), 24.f);
}

TEST(Permute, GPU_Rank3_Values) {
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i);
    auto t = Tensor<float>::from_vector(data, {2, 3, 4}, Device("cuda:0"));

    auto p = t.permute({1, 2, 0});
    auto v = to_host(p);
    EXPECT_FLOAT_EQ(v[0],  0.f);
    EXPECT_FLOAT_EQ(v[13], 18.f);
}
