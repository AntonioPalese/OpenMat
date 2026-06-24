#include "test_helpers.h"

// ---- reshape ----------------------------------------------------------------

TEST(Reshape, FlatTo2D) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {6});
    auto r = t.reshape({2, 3});
    ASSERT_EQ(r.rank(), 2u);
    ASSERT_EQ(r.shape()[0], 2u);
    ASSERT_EQ(r.shape()[1], 3u);
    EXPECT_FLOAT_EQ(r.sum(), 21.0f);
}

TEST(Reshape, TwoDToFlat) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    auto r = t.reshape({6});
    ASSERT_EQ(r.rank(), 1u);
    ASSERT_EQ(r.shape()[0], 6u);
    auto v = to_host(r);
    for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(v[i], static_cast<float>(i + 1));
}

TEST(Reshape, 2DTo3D) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6,7,8,9,10,11,12}, {3,4});
    auto r = t.reshape({2, 2, 3});
    ASSERT_EQ(r.rank(), 3u);
    EXPECT_FLOAT_EQ(r.sum(), 78.0f);
}

TEST(Reshape, PreservesData) {
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<float>(i);
    auto t = Tensor<float>::from_vector(data, {24});
    auto r = t.reshape({2, 3, 4});
    auto v = to_host(r);
    for (int i = 0; i < 24; ++i) EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
}

TEST(Reshape, WrongSizeThrows) {
    auto t = Tensor<float>::from_vector({1,2,3,4}, {4});
    EXPECT_THROW(t.reshape({3, 2}), std::invalid_argument);
}

TEST(Reshape, GPU) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {6}, Device("cuda:0"));
    auto r = t.reshape({2, 3});
    ASSERT_EQ(r.rank(), 2u);
    EXPECT_FLOAT_EQ(r.sum(), 21.0f);
}

TEST(Reshape, IndependentCopy) {
    auto t = Tensor<float>::from_vector({1,2,3,4}, {4});
    auto r = t.reshape({2, 2});
    // modifying original doesn't affect reshape result
    auto v_before = to_host(r);
    EXPECT_FLOAT_EQ(v_before[0], 1.0f);
}

// ---- flatten ----------------------------------------------------------------

TEST(Flatten, Rank2) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3});
    auto f = t.flatten();
    ASSERT_EQ(f.rank(), 1u);
    ASSERT_EQ(f.shape()[0], 6u);
    EXPECT_FLOAT_EQ(f.sum(), 21.0f);
}

TEST(Flatten, Rank3) {
    auto t = Tensor<float>::ones({2, 3, 4});
    auto f = t.flatten();
    ASSERT_EQ(f.rank(), 1u);
    ASSERT_EQ(f.shape()[0], 24u);
    EXPECT_FLOAT_EQ(f.sum(), 24.0f);
}

TEST(Flatten, AlreadyFlat) {
    auto t = Tensor<float>::from_vector({1,2,3}, {3});
    auto f = t.flatten();
    ASSERT_EQ(f.rank(), 1u);
    ASSERT_EQ(f.shape()[0], 3u);
}

TEST(Flatten, GPU) {
    auto t = Tensor<float>::from_vector({1,2,3,4,5,6}, {2,3}, Device("cuda:0"));
    auto f = t.flatten();
    ASSERT_EQ(f.rank(), 1u);
    EXPECT_FLOAT_EQ(f.sum(), 21.0f);
}

// ---- squeeze ----------------------------------------------------------------

TEST(Squeeze, RemovesAxisOfSize1) {
    auto t = Tensor<float>::ones({1, 4, 1});
    auto s = t.squeeze(0);
    ASSERT_EQ(s.rank(), 2u);
    ASSERT_EQ(s.shape()[0], 4u);
    ASSERT_EQ(s.shape()[1], 1u);
}

TEST(Squeeze, LastAxis) {
    auto t = Tensor<float>::ones({3, 1});
    auto s = t.squeeze(1);
    ASSERT_EQ(s.rank(), 1u);
    ASSERT_EQ(s.shape()[0], 3u);
    EXPECT_FLOAT_EQ(s.sum(), 3.0f);
}

TEST(Squeeze, NonUnitAxisThrows) {
    auto t = Tensor<float>::ones({2, 3});
    EXPECT_THROW(t.squeeze(0), std::invalid_argument);
}

TEST(Squeeze, OutOfRangeThrows) {
    auto t = Tensor<float>::ones({3});
    EXPECT_THROW(t.squeeze(2), std::out_of_range);
}

// ---- unsqueeze --------------------------------------------------------------

TEST(Unsqueeze, InsertAtFront) {
    auto t = Tensor<float>::from_vector({1,2,3}, {3});
    auto u = t.unsqueeze(0);
    ASSERT_EQ(u.rank(), 2u);
    ASSERT_EQ(u.shape()[0], 1u);
    ASSERT_EQ(u.shape()[1], 3u);
    EXPECT_FLOAT_EQ(u.sum(), 6.0f);
}

TEST(Unsqueeze, InsertAtEnd) {
    auto t = Tensor<float>::from_vector({1,2,3}, {3});
    auto u = t.unsqueeze(1);
    ASSERT_EQ(u.rank(), 2u);
    ASSERT_EQ(u.shape()[0], 3u);
    ASSERT_EQ(u.shape()[1], 1u);
}

TEST(Unsqueeze, InsertInMiddle) {
    auto t = Tensor<float>::ones({2, 3});
    auto u = t.unsqueeze(1);
    ASSERT_EQ(u.rank(), 3u);
    ASSERT_EQ(u.shape()[0], 2u);
    ASSERT_EQ(u.shape()[1], 1u);
    ASSERT_EQ(u.shape()[2], 3u);
    EXPECT_FLOAT_EQ(u.sum(), 6.0f);
}

TEST(Unsqueeze, OutOfRangeThrows) {
    auto t = Tensor<float>::ones({3});
    EXPECT_THROW(t.unsqueeze(3), std::out_of_range);
}

TEST(Unsqueeze, SqueezeRoundtrip) {
    auto t = Tensor<float>::from_vector({1,2,3,4}, {2,2});
    auto u = t.unsqueeze(1);   // {2,1,2}
    auto s = u.squeeze(1);     // {2,2}
    ASSERT_EQ(s.rank(), 2u);
    EXPECT_FLOAT_EQ(s.sum(), 10.0f);
}
