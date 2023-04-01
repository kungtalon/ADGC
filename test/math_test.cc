#include <iostream>

#include "gtest/gtest.h"

#include "utils/math_utils.h"

TEST(AdgcMathUtilsTest, DoubleGemmEasyTest) {
  // shape_a : [2,2]
  double a[4] = {10, 13, 18, 13};
  // shape_b: [3, 5, 2]
  double b[4] = {8, 6, 3, 7};
  double true_c[4] = {119, 151, 183, 199};

  double *out_c = new double[4];
  memset(out_c, 0, 4 * sizeof(double));
  utils::math::tensor_gemm(4, 4, 4, 2, 2, 2, a, b, out_c);

  for (int ix = 0; ix < 4; ix++) {
    ASSERT_FLOAT_EQ(true_c[ix], out_c[ix])
        << "Mismatched results at index " << ix << "Extra info:\n"
        << utils::array_to_str(2, 2, true_c, "true")
        << utils::array_to_str(2, 2, out_c, "out");
  }

  delete[] out_c;
}

TEST(AdgcMathUtilsTest, DoubleGemmTest) {
  // shape_a : [3, 4, 5]
  double a[60] = {9,  13, 16, 11, 14, 4,  17, 18, 5,  4,  5,  12, 8,  2,  5,
                  15, 10, 9,  5,  4,  12, 6,  15, 5,  9,  17, 8,  2,  17, 17,
                  12, 18, 13, 4,  19, 2,  7,  4,  16, 19, 9,  19, 2,  15, 6,
                  15, 13, 11, 13, 14, 5,  13, 2,  6,  11, 13, 6,  16, 6,  11};
  // shape_b: [3, 5, 2]
  double b[30] = {7,  14, 6, 8,  16, 12, 19, 10, 19, 8,  13, 3, 19, 7, 5,
                  10, 18, 6, 13, 5,  9,  16, 16, 6,  19, 7,  6, 5,  4, 19};
  double true_c[24] = {872, 644, 589, 490, 368, 322, 480, 480,
                       552, 303, 910, 314, 882, 411, 714, 286,
                       537, 461, 686, 726, 371, 411, 597, 595};

  double *out_c = new double[24];
  utils::math::tensor_gemm(60, 30, 24, 4, 2, 5, a, b, out_c);

  for (int ix = 0; ix < 24; ix++) {
    ASSERT_FLOAT_EQ(true_c[ix], out_c[ix])
        << "Mismatched results at index " << ix << "Extra info:\n"
        << utils::array_to_str(3, 6, true_c, "true")
        << utils::array_to_str(3, 6, out_c, "out");
  }

  delete[] out_c;
}

TEST(AdgcMathUtilsTest, GemmLeft2dRight3dTest) {
  // [3 ,2]
  float a[6] = {-9., 7., -3., -2., -1., -5.};
  // [3, 2, 3]
  float b[18] = {5.,  11., 7., -7., -2., 1., 0., 9.,  7.,
                 -2., -1., 5., 9.,  5.,  3., 0., -3., -3.};
  float *c = new float[27];
  float ec[27] = {-94., -113., -56., -1.,  -29., -23., 30., -1., -12.,
                  -14., -88.,  -28., 4.,   -25., -31., 10., -4., -32.,
                  -81., -66.,  -48., -27., -9.,  -3.,  -9., 10., 12.};

  utils::math::tensor_gemm(6, 18, 27, 3, 3, 2, a, b, c);

  for (int ix = 0; ix < 27; ++ix) {
    EXPECT_FLOAT_EQ(ec[ix], c[ix]);
  }

  delete[] c;
}

TEST(AdgcMathUtilsTest, FloatElementAddTest) {
  float a[3] = {1, 2, 3};
  float b[3] = {6, 7, 2};
  float *c = new float[3];
  memset(c, 0, sizeof(float) * 3);

  utils::math::elementwise_add(3, a, b, c);
  float expect[3] = {7, 9, 5};

  for (int ix = 0; ix < 3; ++ix) {
    EXPECT_FLOAT_EQ(expect[ix], c[ix]);
  }

  delete[] c;
}

TEST(AdgcMathUtilsTest, FloatElementSubTest) {
  float a[3] = {1, 2, 3};
  float b[3] = {6, 8, 2};
  float *c = new float[3];
  memset(c, 0, sizeof(float) * 3);

  utils::math::elementwise_add(3, a, b, c, true);
  float expect[3] = {-5, -6, 1};

  for (int ix = 0; ix < 3; ++ix) {
    EXPECT_FLOAT_EQ(expect[ix], c[ix]);
  }

  delete[] c;
}

TEST(AdgcMathUtilsTest, FloatElementMultplyTest) {
  float a[3] = {1, 2, 3};
  float b[3] = {6, 4, -2};
  float *c = new float[3];
  memset(c, 0, sizeof(float) * 3);

  utils::math::elementwise_multiply(3, a, b, c);
  float expect[3] = {6, 8, -6};

  for (int ix = 0; ix < 3; ++ix) {
    EXPECT_FLOAT_EQ(expect[ix], c[ix]);
  }

  delete[] c;
}

TEST(AdgcMathUtilsTest, DoubleElementMultplyTest) {
  double a[5] = {0.5651718, 0.67220126, 0.92264734, 0.53891886, 0.39417962};
  double b[5] = {0.06695532, 0.66330092, 0.86758923, 0.90448946, 0.13699479};
  double *c = new double[5];
  memset(c, 0, sizeof(float) * 5);

  utils::math::elementwise_multiply(5, a, b, c);
  float expect[5] = {0.03784126, 0.44587171, 0.8004789, 0.48744643, 0.05400055};

  for (int ix = 0; ix < 5; ++ix) {
    EXPECT_FLOAT_EQ(expect[ix], c[ix]);
  }

  delete[] c;
}

TEST(AdgcMathUtilsTest, DoubleFillDiagonalTest) {
  double diags[3] = {1, 2, 3};
  /*
        [1, 0, 0, 0]
        [0, 2, 0, 0]
        [0, 0, 3, 0]
  */
  double *zeros = new double[12];
  memset(zeros, 0, sizeof(double) * 12);
  utils::math::fill_diagonal(3, 4, diags, zeros);

  double expected[12] = {1., 0., 0, 0, 0, 2, 0, 0, 0, 0, 3, 0};
  for (int ix = 0; ix < 12; ++ix) {
    EXPECT_FLOAT_EQ(expected[ix], zeros[ix]);
  }

  delete[] zeros;
}

TEST(AdgcMathUtilsTest, Kronecker1DTest) {
  double a[3] = {1., 2., 3.};
  double b[4] = {3, 4, 5, 6};

  double *zeros = new double[13];
  memset(zeros, 0, sizeof(double) * 13);
  utils::math::kron1d(3, 4, 12, a, b, zeros);

  double expected[12] = {3, 4, 5, 6, 6, 8, 10, 12, 9, 12, 15, 18};
  for (int ix = 0; ix < 12; ++ix) {
    EXPECT_FLOAT_EQ(expected[ix], zeros[ix]);
  }

  delete[] zeros;
}

TEST(AdgcMathUtilsTest, Kronecker2DTest) {
  double a[6] = {1., 2., 3., 4., 5., 6.};
  double b[6] = {-1, -2, -3, -4, -5., -6};

  double *zeros = new double[37];
  memset(zeros, 0, sizeof(double) * 37);
  utils::math::tensor_kron_product(6, 6, 3, 2, 2, 3, a, b, zeros);

  double expected[36] = {-1.,  -2.,  -3.,  -2.,  -4.,  -6.,  -4.,  -5.,  -6.,
                         -8.,  -10., -12., -3.,  -6.,  -9.,  -4.,  -8.,  -12.,
                         -12., -15., -18., -16., -20., -24., -5.,  -10., -15.,
                         -6.,  -12., -18., -20., -25., -30., -24., -30., -36.};
  for (int ix = 0; ix < 36; ++ix) {
    EXPECT_FLOAT_EQ(expected[ix], zeros[ix]);
  }

  delete[] zeros;
}

TEST(AdgcMathUtilsTest, SigmoidTest) {
  double a[6] = {1., 2., 3., 4., 5., 6.};

  double *b = new double[6];
  memset(b, 0, sizeof(double) * 6);
  for (int ix = 0; ix < 6; ++ix) {
    b[ix] = utils::math::sigmoid(a[ix]);
  }

  double expected[6] = {0.7310585975646973, 0.8807970285415649,
                        0.9525741338729858, 0.9820137619972229,
                        0.9933071732521057, 0.9975274205207825};
  for (int ix = 0; ix < 6; ++ix) {
    EXPECT_FLOAT_EQ(expected[ix], b[ix]);
  }

  delete[] b;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}