#include "autodiff/component/functional.h"
#include "autodiff/component/ops.h"
#include "autodiff/component/variable.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace auto_diff;

TEST(OpsTest, AddAndReduceSumTest) {

  // this block limits the lifetime of all graph nodes
  try {
    Graph *graph = Graph::get_instanceof_global_graph();

    Variable v1 = Variable({2, 2});
    Variable v2 = Variable({2, 2});

    DTensor value_v1 = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
    DTensor value_v2 = tensor::Tensor<double>({2, 2}, {5, 6, 7, 8});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto add_ops = ops::MatSum({&v1, &v2});
    auto target = functional::ReduceSum(&add_ops);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(v1.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(v2.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(add_ops.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(36.));

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 4);

    for (auto node_ptr : nodes) {
      if (node_ptr->get_type() == NodeType::ADG_VARIABLE_TYPE) {
        node_ptr->backward(&target);
      }
    }

    // test backward
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(36.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(1., 1., 1., 1.));
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAre(1., 1., 1., 1.));

    Graph::delete_global_graph();
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
}

TEST(OpsTest, VecDotTest) {
  // this block limits the lifetime of all graph nodes
  try {
    Graph *graph = Graph::get_instanceof_global_graph();

    Variable v1 = Variable({2, 1});
    Variable v2 = Variable({2, 1});

    DTensor value_v1 = tensor::Tensor<double>({2, 1}, {1, 2});
    DTensor value_v2 = tensor::Tensor<double>({2, 1}, {5, 6});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto target = ops::VecDot(&v1, &v2);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(v1.get_value_shape(), tensor::TensorShape({2, 1}));
    ASSERT_EQ(v2.get_value_shape(), tensor::TensorShape({2, 1}));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(17.));

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 3);

    for (auto node_ptr : nodes) {
      if (node_ptr->get_type() == NodeType::ADG_VARIABLE_TYPE) {
        node_ptr->backward(&target);
      }
    }

    // test backward
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(17.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 1}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(5., 6.));
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAre(1., 2.));

    Graph::delete_global_graph();
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
}

TEST(OpsTest, MatMulTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {

    Variable v1 = Variable({2, 2});
    Variable v2 = Variable({2, 2});

    DTensor value_v1 = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
    DTensor value_v2 = tensor::Tensor<double>({2, 2}, {5, 6, 7, 8});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto matmul_ops = ops::MatMul(&v1, &v2);
    auto target = functional::ReduceSum(&matmul_ops);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(v1.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(v2.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(matmul_ops.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(matmul_ops.get_value().to_int().to_vector(),
                ElementsAre(19, 22, 43, 50));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(134.));

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 4);

    for (auto node_ptr : nodes) {
      if (node_ptr->get_type() == NodeType::ADG_VARIABLE_TYPE) {
        node_ptr->backward(&target);
      }
    }

    // test backward
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(134.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(11., 15., 11., 15.));
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAre(4., 4., 6., 6.));

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

TEST(OpsTest, MatMul3DTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {

    Variable v1 = Variable({3, 2, 3});
    Variable v2 = Variable({3, 2});

    DTensor value_v1 = tensor::Tensor<double>({3, 2, 3}, {5., 11., 7., -7., -2., 1., 0., 9., 7., -2., -1., 5., 9., 5.,
                                                          3., 0., -3., -3.});
    DTensor value_v2 = tensor::Tensor<double>({3, 2}, {-9., 7., -3., -2., -1., -5.});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto matmul_ops = ops::MatMul(&v1, &v2);
    auto target = functional::ReduceSum(&matmul_ops);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(matmul_ops.get_value_shape(), tensor::TensorShape({3, 2, 2}));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(-225.));

    graph->backward(target);

    // test backward
    ASSERT_THAT(target.get_value().to_vector(), ElementsAre(-225.));
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2, 3}));
    ASSERT_THAT(v1.get_grad().to_vector(),
                ElementsAre(-2., -5., -6., -2., -5., -6., -2., -5., -6., -2., -5., -6., -2., -5.,
                            -6., -2., -5., -6.));
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAre(5., 5., 19., 19., 20., 20.));

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

TEST(OpsTest, PointMulReduceMeanTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {

    Variable v1 = Variable({3, 2, 4});
    Variable v2 = Variable({3, 2, 4});

    DTensor value_v1 =
      tensor::Tensor<double>({3, 2, 4}, {2., 5., 7., 7., 1., 9., 7., 8., 3., 1., 3., 8., 4., 8., 5., 7., 2., 4.,
                                         4., 5., 8., 3., 8., 4.});
    DTensor value_v2 =
      tensor::Tensor<double>({3, 2, 4}, {7., 0., 5., 6., 4., 3., 3., 7., 9., 8., 0., 8., 5., 2., 3., 3., 1., 8.,
                                         3., 6., 4., 1., 4., 2.});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto pm = ops::PointMul(&v1, &v2);
    auto target = functional::ReduceMean(&pm);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(pm.get_value_shape(), tensor::TensorShape({3, 2, 4}));
    ASSERT_THAT(pm.get_value().to_vector(),
                ElementsAre(14., 0., 35., 42., 4., 27., 21., 56., 27., 8., 0., 64., 20., 16.,
                            15., 21., 2., 32., 12., 30., 32., 3., 32., 8.));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 21.70833396911621);

    graph->backward(target);

    // test backward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 21.70833396911621);
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2, 4}));
    float v1e[24] =
      {0.2916666865348816, 0.0, 0.2083333432674408, 0.25, 0.1666666716337204, 0.125, 0.125, 0.2916666865348816, 0.375,
       0.3333333432674408, 0.0, 0.3333333432674408, 0.2083333432674408, 0.0833333358168602, 0.125, 0.125,
       0.0416666679084301, 0.3333333432674408, 0.125, 0.25, 0.1666666716337204, 0.0416666679084301, 0.1666666716337204,
       0.0833333358168602};
    auto v1o = v1.get_grad().to_vector();
    for (int ix = 0; ix < 24; ++ix) {
      ASSERT_FLOAT_EQ(v1o[ix], v1e[ix]);
    }
    float v2e[24] =
      {0.0833333358168602, 0.2083333432674408, 0.2916666865348816, 0.2916666865348816, 0.0416666679084301, 0.375,
       0.2916666865348816, 0.3333333432674408, 0.125, 0.0416666679084301, 0.125, 0.3333333432674408, 0.1666666716337204,
       0.3333333432674408, 0.2083333432674408, 0.2916666865348816, 0.0833333358168602, 0.1666666716337204,
       0.1666666716337204, 0.2083333432674408, 0.3333333432674408, 0.125, 0.3333333432674408, 0.1666666716337204};
    auto v2o = v2.get_grad().to_vector();
    for (int ix = 0; ix < 24; ++ix) {
      ASSERT_FLOAT_EQ(v2o[ix], v2e[ix]);
    }
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

TEST(OpsTest, Pad2DTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {

    Variable v1 = Variable({3, 2, 4});

    DTensor value_v1 =
      tensor::Tensor<double>({3, 2, 4}, {3, 16, 18, 3, 16, -2, -10, 5, -4, 16, 4, -9, 13,
                                         -5, 19, 10, 8, -6, -10, -10, 19, -7, 14, 14});

    v1.assign_value(value_v1);

    auto pd = ops::Pad2D(&v1, {{1, 2}, {0, 1}});
    auto target = functional::ReduceMean(&pd);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(pd.get_value_shape(), tensor::TensorShape({3, 5, 5}));
    ASSERT_THAT(pd.get_value().to_vector(),
                ElementsAre(0., 0., 0., 0., 0., 3., 16., 18., 3., 0., 16., -2.,
                            -10., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., -4., 16., 4., -9., 0., 13.,
                            -5., 19., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 8., -6., -10., -10., 0.,
                            19., -7., 14., 14., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0.));
    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 1.5333333333333334);

    graph->backward(target);

    // test backward
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2, 4}));
    float v1e[24] =
      {0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334,
       0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334,
       0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334,
       0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334,
       0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334};
    auto v1o = v1.get_grad().to_vector();
    for (int ix = 0; ix < 24; ++ix) {
      ASSERT_FLOAT_EQ(v1o[ix], v1e[ix]);
    }
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

TEST(FunctionalTest, SigmoidReluTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = new Graph();

  try {
    Variable v1 = Variable({3, 2}, graph);
    Variable v2 = Variable({2, 1}, graph);
    Variable v3 = Variable({1}, graph);

    DTensor value_v1 = tensor::Tensor<double>({3, 2}, {1, 2, 3, 3, 4, 5});
    DTensor value_v2 = tensor::Tensor<double>({2, 1}, {2, 1});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);
    v3.assign_value(DTensor({1}, -8));

    auto matmul_ops = ops::MatMul(&v1, &v2, graph);
    auto add_ops = ops::Add(&matmul_ops, &v3, graph);
    auto relu = functional::ReLU(&add_ops, graph);
    auto sigmoid = functional::Sigmoid(&relu, graph);
    auto target = functional::ReduceSum(&sigmoid, graph);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(matmul_ops.get_value_shape(), tensor::TensorShape({3, 1}));
    ASSERT_THAT(relu.get_value().to_vector(), ElementsAre(0., 1., 5.));
    ASSERT_THAT(sigmoid.get_value().to_vector(),
                ElementsAre(0.5, 0.7310585786300049, 0.9933071490757153));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 2.2243657277057203);

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 8);

    for (auto node_ptr : nodes) {
      if (node_ptr->get_type() == NodeType::ADG_VARIABLE_TYPE) {
        node_ptr->backward(&target);
      }
    }

    // test backward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 2.2243657277057203);
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2}));
    double v1_grad_exp[6] = {0.0,
                             0.0,
                             0.3932238664829637,
                             0.19661193324148185,
                             0.013296113341580066,
                             0.006648056670790033};
    auto v1_grad_out = v1.get_grad().to_vector();
    for (int ix = 0; ix < 2; ++ix) {
      ASSERT_FLOAT_EQ(v1_grad_out[ix], v1_grad_exp[ix]);
    }
    double v2_grad_exp[2] = {0.6164280264076056, 0.6230760830783957};
    auto v2_grad_out = v2.get_grad().to_vector();
    for (int ix = 0; ix < 2; ++ix) {
      ASSERT_FLOAT_EQ(v2_grad_out[ix], v2_grad_exp[ix]);
    }
    ASSERT_FLOAT_EQ(v3.get_grad().get_value(), 0.20325998991227187);

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  delete graph;
}

TEST(FunctionalTest, CrossEntropyWithSoftmaxTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({3, 2});
    Variable v2 = Variable({2, 3});
    Variable v3 = Variable({1});
    Variable label = Variable({3, 3});

    DTensor value_v1 = tensor::Tensor<double>({3, 2}, {1, 2, 3, 3, 4, 5});
    DTensor value_v2 = tensor::Tensor<double>({2, 3}, {10, 2, 3, 4, 5, 9});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);
    v3.assign_value(tensor::Tensor<double>({1}, 2));
    label.assign_value(
      tensor::Tensor<double>({3, 3}, {1, 0, 0, 1, 0, 0, 1, 0, 0}));

    auto matmul_ops = ops::MatMul(&v1, &v2);
    auto add_ops = ops::Add(&matmul_ops, &v3);
    auto target = functional::CrossEntropyWithSoftMax(&add_ops, &label);

    graph->zero_grad();
    target.forward();

    // test forward
    ASSERT_EQ(matmul_ops.get_value_shape(), tensor::TensorShape({3, 3}));
    ASSERT_THAT(matmul_ops.get_value().to_vector(),
                ElementsAre(18., 12., 21., 42., 21., 36., 60., 33., 57.));

    auto softmax_out =
      functional::CrossEntropyWithSoftMax::softmax(add_ops.get_value())
        .to_vector();
    double softmax_expect[9] = {
      0.04742029859017179, 0.00011754316834855699, 0.9524621582414796,
      0.9975273760888543, 7.563811607690144e-10, 0.002472623154764529,
      0.9525741268207277, 1.790390521249113e-12, 0.04742587317748187};
    for (int ix = 0; ix < 9; ++ix) {
      ASSERT_FLOAT_EQ(softmax_expect[ix], softmax_out[ix]);
    }

    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 3.099767939120474);

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 7);

    for (auto node_ptr : nodes) {
      if (node_ptr->get_type() == NodeType::ADG_VARIABLE_TYPE) {
        node_ptr->backward(&target);
      }
    }

    // test backward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 3.099767939120474);
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2}));
    double v1_grad_exp[6] = {-6.668175453037145, 4.7624283343757465,
                             -0.017308368134400752, 0.012363116530203897,
                             -0.33198111225669635, 0.23712936588919964};
    auto v1_grad_out = v1.get_grad().to_vector();
    for (int ix = 0; ix < 6; ++ix) {
      ASSERT_FLOAT_EQ(v1_grad_out[ix], v1_grad_exp[ix]);
    }
    double v2_grad_exp[6] = {-1.1497010658603544, 0.00011754544465360147,
                             1.1495835204157006, -2.1497066404494545,
                             0.00023508861479254903, 2.149471551834662};
    auto v2_grad_out = v2.get_grad().to_vector();
    for (int ix = 0; ix < 6; ++ix) {
      ASSERT_FLOAT_EQ(v2_grad_out[ix], v2_grad_exp[ix]);
    }

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

TEST(FunctionalTest, FunctionStyleTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({3, 2});
    Variable v2 = Variable({2, 3});
    Variable v3 = Variable({3, 3});
    Variable label = Variable({3, 3});

    DTensor value_v1 = tensor::Tensor<double>({3, 2}, {1, 2, 3, 3, 4, 5});
    DTensor value_v2 = tensor::Tensor<double>({2, 3}, {10, 2, 3, 4, 5, 9});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);
    v3.assign_value(tensor::Tensor<double>({3, 3}, 2));
    label.assign_value(
      tensor::Tensor<double>({3, 3}, {1, 0, 0, 1, 0, 0, 1, 0, 0}));

    auto target = functional::cross_entropy_with_softmax(
      functional::relu(ops::matsum(ops::matmul(v1, v2), v3)), label);

    graph->zero_grad();
    target.forward();

    // test forward

    ASSERT_EQ(target.get_value_shape(), tensor::TensorShape({1}));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 3.099767939120474);

    auto nodes = graph->get_node_list();
    ASSERT_EQ(nodes.size(), 8);

    graph->backward(target);

    // test backward
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 3.099767939120474);
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({3, 2}));
    double v1_grad_exp[6] = {-6.668175453037145, 4.7624283343757465,
                             -0.017308368134400752, 0.012363116530203897,
                             -0.33198111225669635, 0.23712936588919964};
    auto v1_grad_out = v1.get_grad().to_vector();
    for (int ix = 0; ix < 6; ++ix) {
      ASSERT_FLOAT_EQ(v1_grad_out[ix], v1_grad_exp[ix]);
    }
    double v2_grad_exp[6] = {-1.1497010658603544, 0.00011754544465360147,
                             1.1495835204157006, -2.1497066404494545,
                             0.00023508861479254903, 2.149471551834662};
    auto v2_grad_out = v2.get_grad().to_vector();
    for (int ix = 0; ix < 6; ++ix) {
      ASSERT_FLOAT_EQ(v2_grad_out[ix], v2_grad_exp[ix]);
    }

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