//
// Created by kungtalon on 2022/12/23.
//
#include "data/dataset.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace auto_diff;

#define MNIST_DATA_PATH "/home/kungtalon/projects/automatic_differentiation_graph_cpp/testdata/test.csv"
#define BATCH_SIZE 3

TEST(DataSetTest, CsvDataSetTest) {
  try {
    auto data_set = data::CsvDataset(MNIST_DATA_PATH, 0, BATCH_SIZE);

    EXPECT_EQ(data_set.get_label_count(), 10);
    EXPECT_EQ(data_set.get_data_count(), 2000);

    tensor::Tensor<double> x, y;
    DataPair data_pair;
    size_t index = 0;
    while (data_set.has_next()) {
      if (index == 0) {
        data_pair = data_set.get_next();
        x = data_pair.first;
        y = data_pair.second;
        EXPECT_EQ(x.get_shape(), std::vector<size_t>({3, 784}));
        EXPECT_EQ(x.get_value({0, 0}), 0.);
        EXPECT_EQ(x.get_value({0, 104}), 10);
        EXPECT_EQ(x.get_value({1, 297}), 254);
        EXPECT_EQ(x.get_value({2, 713}), 231);
        EXPECT_EQ(y.get_value({0}), 4);
        EXPECT_EQ(y.get_value({1}), 9);
        EXPECT_EQ(y.get_value({2}), 9);
        ++index;
        continue;
      }

      if (index == 2) {
        data_pair = data_set.get_next();
        x = data_pair.first;
        y = data_pair.second;
        EXPECT_EQ(x.get_shape(), std::vector<size_t>({3, 784}));
        EXPECT_EQ(x.get_value({0, 0}), 0.);
        EXPECT_EQ(x.get_value({0, 240}), 253);
        EXPECT_EQ(x.get_value({1, 154}), 255);
        EXPECT_EQ(x.get_value({2, 518}), 252);
        EXPECT_EQ(y.get_value({0}), 9);
        EXPECT_EQ(y.get_value({1}), 0);
        EXPECT_EQ(y.get_value({2}), 7);
        ++index;
        continue;
      }

      data_set.get_next();
      ++index;
    }

    EXPECT_EQ(index, 667);
  } catch (
    const std::exception &ex
  ) {
    FAIL()
        << "Failed and got this: " << std::endl << ex.
          what();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}