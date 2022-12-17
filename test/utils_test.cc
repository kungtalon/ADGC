#include "utils/utils.h"

#include "gtest/gtest.h"

TEST(UtilsTest, ArrayPrintTest) {
  // [[15, 17,  5, 13],
  //  [14,  9, 10, 19],
  //  [ 3,  3,  1,  8]]
  int32_t a[12] = {15, 17, 5, 13, 14, 9, 10, 19, 3, 3, 1, 8};
  std::string out = utils::array_to_str(3, 4, a);
  const char *expected = "[ 15\t17\t5\t13 ]\n[ 14\t9\t10\t19 ]\n[ 3\t3\t1\t8 ]";
  ASSERT_STREQ(expected, out.c_str()) << "Mismatch array string:\n"
                                      << "Expect: \n"
                                      << expected << "\nGet: \n"
                                      << out;
}

TEST(UtilsTest, MultiArrayPrint2DTest) {
  // [[15, 17,  5, 13],
  //  [14,  9, 10, 19],
  //  [ 3,  3,  1,  8]]
  float a[12] = {15, 17, 5, 13, 14, 9, 10, 19, 3, 3, 1, 8};
  std::string out = utils::multi_array_to_str({3, 4}, a);
  const char *expected =
      "[ [ 15\t17\t5\t13 ]\n[ 14\t9\t10\t19 ]\n[ 3\t3\t1\t8 ] ]";
  ASSERT_STREQ(expected, out.c_str()) << "Mismatch array string:\n"
                                      << "Expect: \n"
                                      << expected << "\nGet: \n"
                                      << out;
}

TEST(UtilsTest, MultiArrayPrint3DEasyTest) {
  // [[15, 17,  5, 13],
  //  [14,  9, 10, 19],
  //  [ 3,  3,  1,  8]]
  float a[12] = {15, 17, 5, 13, 14, 9, 10, 19, 3, 3, 1, 8};
  std::string out = utils::multi_array_to_str({1, 3, 4}, a);
  const char *expected =
      "[ [ [ 15\t17\t5\t13 ]\n[ 14\t9\t10\t19 ]\n[ 3\t3\t1\t8 ] ] ]";
  ASSERT_STREQ(expected, out.c_str()) << "Mismatch array string:\n"
                                      << "Expect: \n"
                                      << expected << "\nGet: \n"
                                      << out;
}

TEST(UtilsTest, MultiArrayPrint3DTest) {
  // [[[ 5,  6],
  //   [18,  7]],

  //  [[ 9, 19],
  //   [10,  8]],

  //  [[12,  1],
  //   [ 1,  4]]]
  std::string out;
  try {
    float a[12] = {5, 6, 18, 7, 9, 19, 10, 8, 12, 1, 1, 4};
    std::vector<size_t> shape = {3, 2, 2};
    out = utils::multi_array_to_str(shape, a);
  } catch (const std::exception &ex) {
    FAIL() << "Getting exception : " << ex.what();
  }
  const char *expected = "[ [ [ 5\t6 ]\n[ 18\t7 ] ]\n[ [ 9\t19 ]\n[ 10\t8 ] "
                         "]\n[ [ 12\t1 ]\n[ 1\t4 ] ] ]";
  ASSERT_STREQ(expected, out.c_str()) << "Mismatch array string:\n"
                                      << "Expect: \n"
                                      << expected << "\nGet: \n"
                                      << out;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}