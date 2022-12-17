#define TENSOR_TESTING true
#define ENABLE_TENSOR_MULTI_THREAD true

#include "autodiff/tensor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;

TEST(AdgcTensorTest, ConstructorTest) {
  std::vector<float> fa = {1., 2., 3., 4.};
  tensor::Tensor<float> ta({2, 2}, fa);
  tensor::Tensor<float> tb(ta);

  ASSERT_THAT(tb.test_get_tensor(), ElementsAre(1, 2, 3, 4));
}

TEST(AdgcTensorTest, SetterGetterTest) {
  std::vector<float> fa = {1, 2, 3, 4};
  tensor::Tensor<float> ta({2, 2}, fa);

  ta.set_value({0, 1}, 4);

  ASSERT_FLOAT_EQ(ta.get_value({0, 1}), 4);

  EXPECT_THROW(ta.get_value({0, 2}),
               adg_exception::InvalidTensorIndexException);
}

TEST(AdgcTensorTest, ShallowCopyTest) {
  std::vector<float> fa = {1, 2, 3, 4};
  tensor::Tensor<float> ta({2, 2}, fa);
  tensor::Tensor<float> tb = ta;
  tensor::Tensor<float> tc(ta);

  ta.set_value({0, 1}, 4);

  ASSERT_FLOAT_EQ(tb.get_value({0, 1}), 4);
  ASSERT_FLOAT_EQ(tc.get_value({0, 1}), 4);
}

TEST(AdgcTensorTest, DeepCopyTest) {
  std::vector<int32_t> fa = {1, 2, 3, 4};
  tensor::Tensor<int32_t> ta({2, 2}, fa);
  tensor::Tensor<int32_t> tb = ta.copy();

  ta.set_value({0, 1}, 4);

  ASSERT_EQ(ta.get_value({0, 1}), 4);
  ASSERT_EQ(tb.get_value({0, 1}), 2);
}

TEST(AdgcTensorTest, ReshapeTest) {
  std::vector<int32_t> fa = {1, 2, 3, 4};
  tensor::Tensor<int32_t> ta({2, 2}, fa);

  ta.set_value({1, 1}, 5);

  ta.reshape({1, 4});

  ASSERT_EQ(ta.get_value({0, 3}), 5);
  ASSERT_EQ(ta.get_value({0, 1}), 2);
  EXPECT_THROW(ta.get_value({1, 1}),
               adg_exception::InvalidTensorIndexException);

  ta.reshape({4});
  ASSERT_EQ(ta.get_value({2}), 3);
  EXPECT_THROW(ta.get_value({0, 4}),
               adg_exception::InvalidTensorIndexException);
}

TEST(AdgcTensorTest, Transpose2DTest) {
  std::vector<double> fa = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  tensor::Tensor<double> ta({3, 3}, fa);
  tensor::Tensor<double> tb = ta.transpose();

  ASSERT_FLOAT_EQ(tb.get_value({0, 1}), 4);
  ASSERT_FLOAT_EQ(tb.get_value({2, 2}), 9);
}

TEST(AdgcTensorTest, Transpose4DTest1) {
  try {
    std::vector<double> fa = {4, 2, 9, 4, 7, 8, 7, 1, 9, 3, 2, 1,
                              9, 5, 8, 4, 0, 6, 8, 4, 3, 2, 3, 9};
    tensor::Tensor<double> ta({3, 1, 2, 4}, fa);
    tensor::Tensor<double> tb = ta.transpose(0, 3);

    ASSERT_THAT(tb.test_get_tensor(),
                ElementsAre(4., 9., 0., 7., 9., 3., 2., 3., 6., 8., 5., 2., 9.,
                            2., 8., 7., 8., 3., 4., 1., 4., 1., 4., 9.));
    ASSERT_FLOAT_EQ(tb.get_value({0, 0, 1, 1}), 9);
    ASSERT_FLOAT_EQ(tb.get_value({1, 0, 1, 2}), 2);

  } catch (const std::exception &ex) {
    FAIL() << ex.what() << std::endl;
  }
}

TEST(AdgcTensorTest, Transpose4DTest2) {
  try {
    std::vector<double> fa = {4, 2, 9, 4, 7, 8, 7, 1, 9, 3, 2, 1,
                              9, 5, 8, 4, 0, 6, 8, 4, 3, 2, 3, 9};
    tensor::Tensor<double> ta({3, 1, 2, 4}, fa);
    tensor::Tensor<double> tc = ta.transpose(1, 2);

    ASSERT_THAT(tc.test_get_tensor(),
                ElementsAre(4., 2, 9, 4, 7, 8, 7, 1, 9, 3, 2, 1, 9, 5, 8, 4, 0,
                            6, 8, 4, 3, 2, 3, 9));
    ASSERT_FLOAT_EQ(tc.get_value({0, 1, 0, 1}), 8);
    ASSERT_FLOAT_EQ(tc.get_value({2, 1, 0, 2}), 3);
  } catch (const std::exception &ex) {
    FAIL() << ex.what() << std::endl;
  }
}

TEST(AdgcTensorTest, TransposeAfterReshapeTest) {
  try {
    std::vector<double> fa = {
        9.18560879, 6.12390408, 3.17782584, 2.72261349, 3.57568576, 3.76135061,
        7.92799201, 9.98124661, 1.27645199, 5.70451287, 7.5644529,  8.74826167,
        2.02890156, 2.9316532,  1.24171595, 2.81467618, 5.18906807, 1.30958224,
        8.83860768, 7.62692663, 5.26178596, 4.12257506, 8.00484642, 2.31013045,
        9.36853919, 9.56309285, 7.54392209, 9.53950562, 7.25116432, 7.16680406,
        8.83593513, 8.68344079, 8.97369036, 2.52141155, 0.30235258, 1.49508892,
        3.65745596, 7.98818601, 3.73459975, 2.50854687, 4.88304246, 5.19887896,
        6.91659153, 8.18789307, 0.0460029,  5.75502117, 3.60800383, 9.45353803,
        7.77987851, 9.2211933,  3.91954351, 1.22402155, 2.06577014, 4.92628141,
        7.4176446,  6.62866591, 8.77889832, 7.40832239, 1.60547863, 1.00070277,
        9.16268423, 2.20110862, 6.68851616, 0.0866078,  2.50258431, 3.571241,
        2.80895354, 1.59300913, 1.17286671, 0.59429752, 7.6627302,  9.74814692,
        7.56475407, 8.73227613, 4.67892615, 2.29259959, 8.6056778,  6.28754323,
        6.96007614, 5.45848756, 5.75042423, 2.21983488, 1.1271309,  8.35740777,
        6.68311655, 5.8169939,  7.39595171, 3.51530673, 4.81747005, 0.4553131,
        5.43168672, 8.19066175, 3.00689447, 6.70645165, 6.47044776, 6.12177263};
    tensor::Tensor<double> ta({4, 2, 3, 4}, fa);

    ta.reshape({8, 1, 3, 4});

    tensor::Tensor<double> tc = ta.transpose(1, 2);

    ASSERT_THAT(
        tc.test_get_tensor(),
        ElementsAre(9.18560879, 6.12390408, 3.17782584, 2.72261349, 3.57568576,
                    3.76135061, 7.92799201, 9.98124661, 1.27645199, 5.70451287,
                    7.5644529, 8.74826167, 2.02890156, 2.9316532, 1.24171595,
                    2.81467618, 5.18906807, 1.30958224, 8.83860768, 7.62692663,
                    5.26178596, 4.12257506, 8.00484642, 2.31013045, 9.36853919,
                    9.56309285, 7.54392209, 9.53950562, 7.25116432, 7.16680406,
                    8.83593513, 8.68344079, 8.97369036, 2.52141155, 0.30235258,
                    1.49508892, 3.65745596, 7.98818601, 3.73459975, 2.50854687,
                    4.88304246, 5.19887896, 6.91659153, 8.18789307, 0.0460029,
                    5.75502117, 3.60800383, 9.45353803, 7.77987851, 9.2211933,
                    3.91954351, 1.22402155, 2.06577014, 4.92628141, 7.4176446,
                    6.62866591, 8.77889832, 7.40832239, 1.60547863, 1.00070277,
                    9.16268423, 2.20110862, 6.68851616, 0.0866078, 2.50258431,
                    3.571241, 2.80895354, 1.59300913, 1.17286671, 0.59429752,
                    7.6627302, 9.74814692, 7.56475407, 8.73227613, 4.67892615,
                    2.29259959, 8.6056778, 6.28754323, 6.96007614, 5.45848756,
                    5.75042423, 2.21983488, 1.1271309, 8.35740777, 6.68311655,
                    5.8169939, 7.39595171, 3.51530673, 4.81747005, 0.4553131,
                    5.43168672, 8.19066175, 3.00689447, 6.70645165, 6.47044776,
                    6.12177263));
  } catch (const std::exception &ex) {
    FAIL() << ex.what() << std::endl;
  }
}

TEST(AdgcTensorTest, TensorMultiplyTest) {
  std::vector<double> fa = {
      0.06070769, 0.73364242, 0.92237306, 0.54659584, 0.64786345, 0.42514575,
      0.99104248, 0.61900974, 0.83586207, 0.40331749, 0.5661589,  0.88342392,
      0.72669169, 0.09861136, 0.19608444, 0.82315503, 0.51640939, 0.02420425,
      0.25253292, 0.45016243, 0.38296859, 0.16409721, 0.84429582, 0.40846548,
      0.35060328, 0.46719049, 0.4388522,  0.27731552, 0.7296538,  0.88894036,
      0.86651689, 0.17049278, 0.91408396, 0.63191992, 0.63814401, 0.625566};
  tensor::Tensor<double> ta({3, 2, 2, 3}, fa);

  std::vector<double> fb = {
      0.85469578, 0.55758443, 0.75128807, 0.62902494, 0.86160297, 0.18742677,
      0.04695074, 0.80488391, 0.8560711,  0.87828848, 0.61542361, 0.0606644,
      0.90440609, 0.16065556, 0.09286372, 0.89038293, 0.07723816, 0.33846552,
      0.4155942,  0.07222199, 0.69102907, 0.72036471, 0.56979699, 0.74802183,
      0.2468946,  0.29613471, 0.29508749, 0.21224456, 0.63487395, 0.0953371,
      0.26688447, 0.01662648, 0.00636381, 0.24243166, 0.42455583, 0.48405301};
  tensor::Tensor<double> tb({3, 2, 2, 3}, fb);

  std::vector<double> true_res = {
      0.0518866,  0.40906759, 0.69296788, 0.34382241, 0.55820107, 0.0796837,
      0.04653018, 0.49823098, 0.71555735, 0.3542291,  0.34842756, 0.05359239,
      0.65722439, 0.01584246, 0.01820913, 0.73292319, 0.03988651, 0.00819231,
      0.10495122, 0.03251162, 0.26464243, 0.11820984, 0.48107722, 0.30554109,
      0.08656206, 0.13835132, 0.12949979, 0.05885871, 0.46323819, 0.08474899,
      0.2312599,  0.0028347,  0.00581705, 0.1531974,  0.27092776, 0.30280711};

  auto tc = ta.multiply(tb);
  ASSERT_THAT(tc.get_shape(), ElementsAre(3, 2, 2, 3));

  auto tc_tensor = tc.test_get_tensor();
  ASSERT_EQ(tc_tensor.size(), true_res.size());

  for (int i = 0; i < tc_tensor.size(); ++i) {
    ASSERT_LE(abs(tc_tensor[i] - true_res[i]), 1e-7);
  }

  auto test = [&]() {
    tc.reshape({3, 2, 6});
    ta.multiply(tc);
  };

  EXPECT_THROW(test(), adg_exception::MismatchTensorShapeError);
}

TEST(AdgcTensorTest, TensorDotTest) {
  std::vector<float> fa = {9,  13, 16, 11, 14, 4,  17, 18, 5,  4,  5,  12,
                           8,  2,  5,  15, 10, 9,  5,  4,  12, 6,  15, 5,
                           9,  17, 8,  2,  17, 17, 12, 18, 13, 4,  19, 2,
                           7,  4,  16, 19, 9,  19, 2,  15, 6,  15, 13, 11,
                           13, 14, 5,  13, 2,  6,  11, 13, 6,  16, 6,  11};
  tensor::Tensor<float> ta({3, 4, 5}, fa);

  std::vector<float> fb = {7,  14, 6,  8, 16, 12, 19, 10, 19, 8,
                           13, 3,  19, 7, 5,  10, 18, 6,  13, 5,
                           9,  16, 16, 6, 19, 7,  6,  5,  4,  19};
  tensor::Tensor<float> tb({3, 5, 2}, fb);

  std::vector<float> true_res = {872, 644, 589, 490, 368, 322, 480, 480,
                                 552, 303, 910, 314, 882, 411, 714, 286,
                                 537, 461, 686, 726, 371, 411, 597, 595};

  auto tc = ta.dot(tb);
  ASSERT_THAT(tc.get_shape(), ElementsAre(3, 4, 2));

  auto tc_tensor = tc.test_get_tensor();
  ASSERT_EQ(tc_tensor.size(), true_res.size());

  for (int i = 0; i < tc_tensor.size(); ++i) {
    ASSERT_FLOAT_EQ(tc_tensor[i], true_res[i]);
  }

  EXPECT_THROW(ta.dot(tc), adg_exception::MismatchTensorShapeError);
}

TEST(AdgcTensorTest, TensorAddTest) {
  std::vector<float> fa = {1, 2, 3, 4};
  tensor::Tensor<float> ta({2, 2}, fa);

  std::vector<float> fb = {6, 1, 9, 0};
  tensor::Tensor<float> tb({2, 2}, fb);

  auto tc = ta.add(tb);
  ASSERT_THAT(tc.test_get_tensor(), ElementsAre(7, 3, 12, 4));

  auto test = [&]() {
    tc.reshape({1, 4});
    ta.add(tc);
  };

  EXPECT_THROW(test(), adg_exception::MismatchTensorShapeError);
}

TEST(AdgcTensorTest, TensorMultiplyNumberTest) {
  std::vector<float> fa = {1, 2, 3, 4};
  tensor::Tensor<float> ta({2, 2}, fa);

  auto tc = ta.multiply(3.);
  ASSERT_THAT(tc.test_get_tensor(), ElementsAre(3., 6., 9., 12.));
}

TEST(AdgcTensorTest, TensorAddNumberTest) {
  std::vector<float> fa = {1, 2, 3, 4};
  tensor::Tensor<float> ta({2, 2}, fa);

  auto tc = ta.add(3.);
  ASSERT_THAT(tc.test_get_tensor(), ElementsAre(4., 5., 6., 7.));
}

TEST(AdgcTensorTest, TensorInitTest) {
  tensor::Tensor<float> ta({2, 2});

  // ta.normal_init();

  ta.normal_init(1, 10);

  // FAIL() << ta.to_string();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}