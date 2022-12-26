#include "autodiff/layer/layer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace auto_diff;

TEST(LayerTest, DenseLayerTest) {
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({2, 2});

    DTensor value_v1 = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
    DTensor weight = tensor::Tensor<double>({2, 1}, {-1, 2});
    DTensor bias = tensor::Tensor<double>({1}, 2);

    layer::Dense dense_layer(2, 1);
    dense_layer.assign_weight(weight);
    dense_layer.assign_bias(bias);

    // build graph
    auto target = functional::reduce_sum(dense_layer(v1));

    // forward
    v1.assign_value(value_v1);
    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(v1.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(dense_layer.get_weight().get_value_shape(),
              tensor::TensorShape({2, 1}));
    ASSERT_THAT(dense_layer.get_weight().get_value().to_vector(),
                ElementsAre(-1., 2.));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 12.);

    graph->backward(target);

    // test backward
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(12.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(-1, 2, -1., 2.));
    ASSERT_THAT(dense_layer.get_weight().get_grad().to_vector(),
                ElementsAre(4., 6.));
    ASSERT_FLOAT_EQ(dense_layer.get_bias().get_grad().get_value(), 2.);

    // test again
    v1.assign_value(value_v1);
    graph->zero_grad();
    target.forward();
    graph->backward(target);

    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(12.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(-1, 2, -1., 2.));
    ASSERT_THAT(dense_layer.get_weight().get_grad().to_vector(),
                ElementsAre(4., 6.));
    ASSERT_FLOAT_EQ(dense_layer.get_bias().get_grad().get_value(), 2.);
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  graph->remove_all();
  Graph::delete_global_graph();
}

TEST(LayerTest, ConvLayerTest1) {
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({2, 1, 7, 7});

    DTensor value_v1 = tensor::Tensor<double>({2, 1, 7, 7}, {1., 1., 8., 9., 5., -3., 1., 7., 0., -4., 0., -6.,
                                                             6., -3., 7., 9., 6., 8., 9., -9., 1., 5., -3., 0.,
                                                             -9., 8., 9., -8., -1., -8., -3., -4., -2., -9., -9., -9.,
                                                             -8., -7., 1., -9., 6., 1., -8., 9., 1., 7., 1., -10.,
                                                             3., 0., -1., -1., 7., 5., 9., -2., -8., 1., -7., 5.,
                                                             5., 9., -1., 8., 4., -3., -7., -9., -1., -4., 6., -8.,
                                                             6., -10., 8., -4., -4., 2., -7., -1., -5., 4., -5., -2.,
                                                             2., 2., -7., 8., 8., 6., -3., -2., 8., 8., -8., -10.,
                                                             9., -9.});
    DTensor weight = tensor::Tensor<double>({4, 1, 3, 3}, {1., -3., 7., 5., -2., -9., -4., 8., -8., 8., -10., -8.,
                                                           -10., 8., 7., 5., 0., -3., -4., 1., -2., 6., -9., 1.,
                                                           -7., -1., -4., 8., 4., -10., -7., 6., -8., 3., 5., -4.});
    DTensor bias = tensor::Tensor<double>({4}, {-5., 7., -7., -3.});

    layer::Conv2D conv_layer(1, 4, {3, 3}, {1, 1}, "VALID", "none");
    conv_layer.assign_weight(weight);
    conv_layer.assign_bias(bias);

    // build graph
    auto target = functional::reduce_sum(conv_layer(v1));

    // forward
    v1.assign_value(value_v1);
    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 1224.);

    graph->backward(target);

    // test backward
    std::vector<double> v1_grad_exp({13., 5., -8., -8., -8., -21., -13., 7., 2., -20., -20., -20.,
                                     -27., -22., 4., 11., -30., -30., -30., -34., -41., 4., 11., -30.,
                                     -30., -30., -34., -41., 4., 11., -30., -30., -30., -34., -41., -9.,
                                     6., -22., -22., -22., -13., -28., -3., 9., -10., -10., -10., -7.,
                                     -19., 13., 5., -8., -8., -8., -21., -13., 7., 2., -20., -20.,
                                     -20., -27., -22., 4., 11., -30., -30., -30., -34., -41., 4., 11.,
                                     -30., -30., -30., -34., -41., 4., 11., -30., -30., -30., -34., -41.,
                                     -9., 6., -22., -22., -22., -13., -28., -3., 9., -10., -10., -10.,
                                     -7., -19.});
    std::vector<double> weight_grad_exp({37., 12., -7., -16., -27., -41., -3., -21., -53., 37., 12., -7.,
                                         -16., -27., -41., -3., -21., -53., 37., 12., -7., -16., -27., -41.,
                                         -3., -21., -53., 37., 12., -7., -16., -27., -41., -3., -21., -53.});
    std::vector<double> bias_grad_exp({50., 50., 50., 50.});
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAreArray(v1_grad_exp));
    ASSERT_THAT(conv_layer.get_weight().get_grad().to_vector(),
                ElementsAreArray(weight_grad_exp));
    ASSERT_THAT(conv_layer.get_bias().get_grad().to_vector(), ElementsAreArray(bias_grad_exp));

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  graph->remove_all();
  Graph::delete_global_graph();
}

TEST(LayerTest, ConvLayerTest2) {
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({2, 3, 9, 9});

    DTensor value_v1 = tensor::Tensor<double>({2, 3, 9, 9}, {-6., 3., -8., 9., -1., 8., -1., 3., 8., 4., -8., -7.,
                                                             2., 0., -6., -10., 8., -9., 6., 0., -9., -8., 7., -9.,
                                                             5., -5., 5., -6., 6., 7., -3., 1., -9., -5., 4., -6.,
                                                             0., -2., 7., 1., -9., 7., -9., 7., -4., 4., 0., -5.,
                                                             8., -10., -7., 4., -10., 0., -6., 3., 9., 9., -1., -3.,
                                                             6., -5., 0., 8., -10., -8., 7., -9., -6., -3., -6., -9.,
                                                             2., -5., 1., -7., 4., -1., 9., 0., -7., 5., -5., 6.,
                                                             3., -5., 3., 9., 8., -5., 8., -2., -2., 8., 9., -5.,
                                                             -8., -8., 8., -6., 9., 3., -2., -4., -9., 4., -2., 5.,
                                                             -6., 7., 1., 6., 6., 0., 3., -7., 2., 7., -8., 4.,
                                                             -8., -4., 8., -1., -6., 2., -3., -10., -1., -8., 1., 6.,
                                                             9., -9., 0., 3., 8., -2., -4., 4., -3., 6., 6., -10.,
                                                             3., 5., -4., 1., -6., -2., 8., -1., -9., -7., -7., -7.,
                                                             8., -8., 4., 9., -3., -1., -7., 8., 4., 0., 1., -1.,
                                                             4., 6., 7., 0., 8., -8., 0., -9., 2., -5., -1., -7.,
                                                             0., 4., 7., -2., -4., -3., 1., -2., 0., 4., -5., -6.,
                                                             5., 9., -1., 8., -4., 1., -2., 0., -2., 4., -5., -3.,
                                                             9., -5., 3., -4., -7., -4., -10., -5., -1., -9., -4., 4.,
                                                             -5., 3., 6., 7., 4., 7., -9., 4., -9., -2., -5., -6.,
                                                             8., -7., 5., 5., -5., 5., -5., -7., -3., -4., 7., 7.,
                                                             7., -7., -2., 1., 2., -2., 8., -8., 6., 2., -6., -5.,
                                                             9., -8., 3., -1., 8., 2., 5., -3., 7., 9., -2., 7.,
                                                             -2., -1., -1., -3., -10., -5., 7., 6., -4., -1., -5., 2.,
                                                             9., 4., -10., -8., 2., 3., -9., -2., 6., -4., -8., -10.,
                                                             0., -6., 9., 4., -2., -10., 3., -1., -6., -2., 8., -5.,
                                                             1., 4., 5., -4., 0., -2., 3., 8., 5., -8., 0., -5.,
                                                             0., -6., -7., -9., -5., 6., -10., -8., -2., -10., -3., -5.,
                                                             -10., 0., -4., 7., 0., 1., 3., 2., -7., 6., -5., 0.,
                                                             4., 2., 0., -1., 1., -8., -9., 2., 3., -10., -7., -1.,
                                                             -2., -1., -5., 8., 8., 2., -3., -9., 3., 9., -1., 0.,
                                                             8., 2., 9., 7., -1., 5., -3., -2., -7., -2., 7., -6.,
                                                             9., -8., 3., -6., 6., 4., 5., -8., -2., -4., -10., -9.,
                                                             5., 4., 9., 1., -5., -3., -5., 4., -5., -6., -8., 3.,
                                                             9., 9., -6., 5., 7., 6., -6., 9., 4., 3., 3., 1.,
                                                             4., 9., -5., -9., -4., 1., 0., 8., -4., -4., 7., 7.,
                                                             0., 4., -2., -3., 8., -9., -2., -5., -2., -10., -8., 0.,
                                                             4., 9., 9., -1., -1., -10., -2., -7., 0., 8., -10., -2.,
                                                             5., 2., 7., 5., 0., -8., -10., 7., -6., -8., -8., -6.,
                                                             2., -1., -2., -3., 0., -7., -2., 3., -3., 8., 8., 7.,
                                                             8., -6., 5., 6., 9., 9., 3., -5., -6., -3., 5., -6.,
                                                             -3., 3., 2., 7., -6., 4.});
    DTensor weight = tensor::Tensor<double>({5, 3, 2, 2}, {-10., -6., 8., 6., -6., -8., -6., -3., 2., 6., -2., 0.,
                                                           5., 3., 8., -9., -8., 8., -3., 7., 2., 1., 8., 5.,
                                                           7., -4., 7., 9., -2., 7., 7., 0., -8., 6., -2., 7.,
                                                           -2., 6., -1., -5., 4., 9., 7., 1., 4., -8., -9., -4.,
                                                           -6., 0., 1., -1., 1., -9., 8., 2., 8., -1., 8., 8.});
    DTensor bias = tensor::Tensor<double>({5}, {0., 0., -2., -6., 1.});

    layer::Conv2D conv_layer(3, 5, {2, 2}, {2, 2}, {1, 1});
    conv_layer.assign_weight(weight);
    conv_layer.assign_bias(bias);

    // build graph
    auto target = functional::reduce_sum(conv_layer(v1));

    // forward
    v1.assign_value(value_v1);
    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 10783.);

    graph->backward(target);

    // test backward
    std::vector<double> v1_grad_exp({-14., 9., -10., 23., 0., 23., 0., 16., 14., 2., -3., 9.,
                                     -2., 6., -16., -6., 10., 5., 4., 8., -15., -1., -5., 9.,
                                     5., 14., -5., -6., -2., 6., -5., -3., -6., -1., 2., -7.,
                                     5., -1., -5., 16., -3., 23., 0., 23., 6., 9., -9., -10.,
                                     2., -7., -2., 6., -10., -6., -15., 16., 14., 23., 6., -1.,
                                     -5., 8., 6., 6., -10., -6., -1., 2., -4., -7., -10., -6.,
                                     -5., 8., 6., 7., 3., 24., 5., 8., 6., 8., 5., 9.,
                                     13., 7., 13., 7., 9., -1., 16., -3., 8., 4., 9., -5.,
                                     -17., -6., 24., 1., 12., 10., 7., 1., 2., -1., 11., 8.,
                                     -17., 4., 9., -14., 0., -11., 7., -16., 7., -1., 7., 1.,
                                     -9., 4., 13., 7., -2., 4., 8., -7., -10., -16., 7., 4.,
                                     9., -6., -8., 10., 9., -1., -2., 4., 7., 1., -6., -3.,
                                     9., -6., -8., 3., 7., -15., -2., -6., -8., 1., -6., -3.,
                                     22., 3., 6., 6., -6., -3., 1., 16., 13., 3., 16., 3.,
                                     16., 4., 15., -2., 14., -8., 4., -8., 10., 5., -2., -1.,
                                     3., 7., 9., -9., -4., 6., 8., -3., 8., 5., 4., -8.,
                                     4., 7., 8., 4., -4., 13., 8., -9., -4., 6., 5., 3.,
                                     16., 4., 12., -8., 2., 11., -4., 13., 4., -8., 2., 6.,
                                     9., 4., 15., 4., 12., -9., -4., -2., 0., -8., 2., 6.,
                                     4., -3., 4., 12., 2., 6., -4., -2., 0., -3., 11., 12.,
                                     20., -2., 0., 14., 17., -4., 15., -6., 14., 10., -1., -5.,
                                     2., -18., 0., -10., -6., -2., 6., -8., 6., 4., 8., 0.,
                                     8., 6., -1., -5., 0., -6., 5., 5., 2., -6., 0., 10.,
                                     5., 0., 0., -6., 6., 4., 1., -1., 14., -5., 0., 0.,
                                     9., -18., 0., -3., -10., -11., -3., 6., -1., -15., 8., 0.,
                                     15., 15., 17., -4., 16., -1., 9., 7., -4., -3., 9., -11.,
                                     -3., 4., 5., -14., 7., 9., 8., -15., 17., -4., 15., -6.,
                                     -1., -1., 6., 19., 10., 8., -2., 7., 1., 16., -1., -8.,
                                     -6., -8., 4., 9., 5., 0., 1., 9., 0., -6., -3., 7.,
                                     1., 15., 3., 15., 2., 16., 1., -9., -6., 24., 0., 0.,
                                     10., 14., 1., 8., 2., 11., 8., 0., 0., 8., -1., -8.,
                                     -8., -1., -13., -9., -9., 6., 10., 9., 0., 1., -3., -1.,
                                     6., 12., 9., 17., -2., 7., -3., 8., -13., -9., -5., 15.,
                                     8., 7., 0., 12., 10., -1., 6., 19., 10., 15., 14., 13.,
                                     5., 16., -13., 3., -9., -4., -2., 14., -3., 2., 6., 4.,
                                     -8., 12., -9., 3., -3., 4., -2., 0., -9., -4., -1., 4.,
                                     -2., -4., -2., 8., -1., -2., -1., 0., 0., 16., -11., 3.,
                                     8., 8., -3., 8., 0., 0., -8., 14., -3., -6., 12., 12.,
                                     6., 2., 6., 9., -3., 4., -4., 7., 14., 13., 14., 20.,
                                     -7., -8., 6., 14., -8., 12., 6., 6., -2., 1., -2., 7.,
                                     7., 9., 14., 13., 5., 16.});
    std::vector<double> weight_grad_exp({-64., -40., 43., -8., -18., -32., -19., -28., -18., 10.,
                                         6., 13., -34., -26., 44., -102., -27., 38., 29., 70.,
                                         -5., 5., 31., 68., 25., -32., 48., 23., -21., 44.,
                                         63., -3., -31., -9., 9., 67., -22., 38., -3., -36.,
                                         13., 46., 39., 24., 34., -44., -20., -20., -70., -30.,
                                         3., -52., -27., -33., 59., 33., 36., -24., 27., 79.});
    std::vector<double> bias_grad_exp({23., 23., 22., 28., 26.});
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAreArray(v1_grad_exp));
    ASSERT_THAT(conv_layer.get_weight().get_grad().to_vector(),
                ElementsAreArray(weight_grad_exp));
    ASSERT_THAT(conv_layer.get_bias().get_grad().to_vector(), ElementsAreArray(bias_grad_exp));

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  graph->remove_all();
  Graph::delete_global_graph();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}