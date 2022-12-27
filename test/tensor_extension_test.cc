//
// Created by kungtalon on 2022/12/27.
//

#include "tensor/tensor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;

TEST(AdgcTensorExtensionTest, PadTest) {
  tensor::Tensor<float> ta({3, 2}, {5, 2, 0, 3, 1, 4});

  tensor::Tensor<float> result1 = tensor::pad2d(ta, {2, 2});

  ASSERT_THAT(result1.to_vector(), ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0, 3,
                                               0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            << "Result is : \n" << result1.to_string();

  tensor::Tensor<float> tb({2, 2, 2}, {4, 6, 8, 1, 4, 5, 2, 4});

  tensor::Tensor<float> result2 = tensor::pad2d(tb, {{1, 2}, {1, 2}});

  ASSERT_THAT(result2.to_vector(), ElementsAre(0, 0, 0, 0, 0, 0, 4, 6, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0))
            << "Result is : \n" << result2.to_string();
}

TEST(AdgcTensorExtensionTest, DilateTest) {
  tensor::Tensor<float> ta({3, 2}, {5, 2, 0, 3, 1, 4});

  tensor::Tensor<float> result1 = tensor::dilate2d(ta, {1, 2});

  ASSERT_EQ(result1.get_shape(), tensor::TensorShape({5, 4}));
  ASSERT_THAT(result1.to_vector(), ElementsAre(5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 4))
            << "Result is : \n" << result1.to_string();

  tensor::Tensor<float> tb({3, 3, 4}, {4, 6, 4, 2, 8, 8, 7, 9, 4, 2, 6, 7, 6, 5, 3, 7, 9, 3, 3, 6, 1, 3,
                                       1, 3, 3, 1, 9, 4, 9, 2, 9, 9, 6, 6, 2, 7});

  tensor::Tensor<float> result2 = tensor::dilate2d(tb, {1, 1});

  std::vector<float> expect2 = {4, 0, 6, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 7, 0, 9, 0,
                                0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 6, 0, 7, 6, 0, 5, 0, 3, 0, 7, 0, 0,
                                0, 0, 0, 0, 0, 9, 0, 3, 0, 3, 0, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3,
                                0, 1, 0, 3, 3, 0, 1, 0, 9, 0, 4, 0, 0, 0, 0, 0, 0, 0, 9, 0, 2, 0,
                                9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 2, 0, 7};
  ASSERT_THAT(result2.to_vector(), ElementsAreArray(expect2))
            << "Result is : \n" << result2.to_string();

  tensor::Tensor<double> result3 = tensor::dilate2d(tb.to_double(), {0, 1});

  std::vector<double> expect3 = {4, 0, 6, 0, 4, 0, 2, 8, 0, 8, 0, 7, 0, 9, 4, 0, 2, 0, 6, 0, 7, 6,
                                 0, 5, 0, 3, 0, 7, 9, 0, 3, 0, 3, 0, 6, 1, 0, 3, 0, 1, 0, 3, 3, 0,
                                 1, 0, 9, 0, 4, 9, 0, 2, 0, 9, 0, 9, 6, 0, 6, 0, 2, 0, 7};
  ASSERT_THAT(result3.to_vector(), ElementsAreArray(expect3))
            << "Result is : \n" << result3.to_string();

}

TEST(AdgcTensorExtensionTest, ReverseTest) {
  tensor::Tensor<float> ta({3, 2}, {5, 2, 0, 3, 1, 4});

  tensor::reverse(ta, 0);

  ASSERT_THAT(ta.to_vector(), ElementsAre(1, 4, 0, 3, 5, 2))
            << "Result is : \n" << ta.to_string();

  tensor::Tensor<float> tb({3, 3, 4}, {4, 6, 4, 2, 8, 8, 7, 9, 4, 2, 6, 7, 6, 5, 3, 7, 9, 3, 3, 6, 1, 3,
                                       1, 3, 3, 1, 9, 4, 9, 2, 9, 9, 6, 6, 2, 7});

  tensor::reverse(tb, 2);

  std::vector<float> expect2 = {2, 4, 6, 4, 9, 7, 8, 8, 7, 6, 2, 4, 7, 3, 5, 6, 6, 3, 3, 9, 3, 1,
                                3, 1, 4, 9, 1, 3, 9, 9, 2, 9, 7, 2, 6, 6};
  ASSERT_THAT(tb.to_vector(), ElementsAreArray(expect2))
            << "Result is : \n" << tb.to_string();

  tensor::Tensor<float> tc({6, 5}, {-6, 6, -7, 7, 1, 7, 2, 3, -3, -2, -1, -8, 1,
                                    -2, -1, -3, 5, -8, -9, 3, -10, -5, 7, -1, -10, -1,
                                    -4, -5, 4, 0});
  tensor::reverse(tc, 1);

  std::vector<float> expect3 = {1, 7, -7, 6, -6, -2, -3, 3, 2, 7, -1, -2, 1,
                                -8, -1, 3, -9, -8, 5, -3, -10, -1, 7, -5, -10, 0,
                                4, -5, -4, -1};
  ASSERT_THAT(tc.to_vector(), ElementsAreArray(expect3))
            << "Result is : \n" << tb.to_string();
}

TEST(AdgcTensorExtensionTest, AddVecTest) {
  std::vector<float> fa = {9, 13, 16, 11, 14, 4, 17, 18, 5, 4, 5, 12,
                           8, 2, 5, 15, 10, 9, 5, 4, 12, 6, 15, 5,
                           9, 17, 8, 2, 17, 17, 12, 18, 13, 4, 19, 2,
                           7, 4, 16, 19, 9, 19, 2, 15, 6, 15, 13, 11,
                           13, 14, 5, 13, 2, 6, 11, 13, 6, 16, 6, 11};
  tensor::Tensor<float> ta({3, 4, 5}, fa);

  tensor::Tensor<float> tb({5}, {5, 4, 3, 2, 1});

  auto res = tensor::add_vec(ta, tb, 2);

  ASSERT_THAT(res.to_vector(), ElementsAre(14, 17, 19, 13, 15, 9, 21, 21, 7, 5, 10, 16, 11, 4, 6, 20, 14,
                                           12, 7, 5, 17, 10, 18, 7, 10, 22, 12, 5, 19, 18, 17, 22, 16, 6,
                                           20, 7, 11, 7, 18, 20, 14, 23, 5, 17, 7, 20, 17, 14, 15, 15, 10,
                                           17, 5, 8, 12, 18, 10, 19, 8, 12));

  tensor::Tensor<float> tc({3}, {1, 2, 3});

  auto res2 = tensor::add_vec(ta, tc, 0);

  ASSERT_THAT(res2.to_vector(), ElementsAre(10, 14, 17, 12, 15, 5, 18, 19, 6, 5, 6, 13, 9, 3, 6, 16, 11,
                                            10, 6, 5, 14, 8, 17, 7, 11, 19, 10, 4, 19, 19, 14, 20, 15, 6,
                                            21, 4, 9, 6, 18, 21, 12, 22, 5, 18, 9, 18, 16, 14, 16, 17, 8,
                                            16, 5, 9, 14, 16, 9, 19, 9, 14));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
