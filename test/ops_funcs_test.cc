#include "autodiff/component/functional.h"
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

    auto add_ops = functional::MatSum({&v1, &v2});
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

    auto target = functional::VecDot(&v1, &v2);

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

    auto matmul_ops = functional::MatMul(&v1, &v2);
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

    auto matmul_ops = functional::MatMul(&v1, &v2);
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

    auto pm = functional::PointMul(&v1, &v2);
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

    auto pd = functional::Pad2D(&v1, {{1, 2}, {0, 1}});
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

    auto matmul_ops = functional::MatMul(&v1, &v2, graph);
    auto add_ops = functional::Add(&matmul_ops, &v3, graph);
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

    auto matmul_ops = functional::MatMul(&v1, &v2);
    auto add_ops = functional::Add(&matmul_ops, &v3);
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
      functional::relu(functional::matsum(functional::matmul(v1, v2), v3)), label);

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

TEST(OpsTest, Conv2dTest) {
  // this block limits the lifetime of all graph nodes
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v0 = Variable({1, 1, 3, 3});
    Parameter p0 = Parameter({1, 1, 2, 2});

    v0.assign_value({{1, 1, 3, 3}, {1, 3, 5, 2, 4, 6, 0, 9, 8}});
    p0.assign_value({{1, 1, 2, 2}, {7, 3, 5, -1}});

    auto vp = functional::Conv2D(&v0, &p0, {1, 1});
    auto s = functional::ReduceSum(&vp);

    graph->zero_grad();
    s.forward();
    graph->backward(s);

    ASSERT_EQ(s.get_value().get_value(), 172.);
    ASSERT_THAT(v0.get_grad().to_vector(), ElementsAre(7., 10., 3., 12., 14., 2., 5., 4., -1.))
              << v0.get_grad().to_string();
    ASSERT_THAT(p0.get_grad().to_vector(), ElementsAre(10., 18., 15., 27.))
              << p0.get_grad().to_string();

    Variable v1 = Variable({3, 4, 8, 10}); // [B, C, H, W]
    Parameter v2 = Parameter({6, 4, 2, 2}); // [c_out, c_in, kh, kw]

    DTensor value_v1 = tensor::Tensor<double>({3, 4, 8, 10},
                                              {-8., 2., 0., 4., -15., -5., -10., -6., 6., 11., -7., 7.,
                                               -1., -5., -5., -9., 1., 2., 13., 8., -12., -14., -15., -15.,
                                               -5., -14., 12., 6., 6., 1., -3., -7., -15., -8., 9., -11.,
                                               3., -9., -3., -1., -7., -9., -10., -3., -5., 12., 2., -14.,
                                               11., -13., -8., 1., -5., -7., 14., 5., 4., -14., 2., 10.,
                                               -9., -4., -10., 8., 13., 3., -12., 9., -4., -1., -8., -2.,
                                               6., 13., 3., 13., 1., -4., 1., -1., -15., 9., -11., 5.,
                                               10., -9., -10., -1., -4., -14., -4., -11., -15., -5., 14., -11.,
                                               -3., -5., -10., 7., 3., -4., 12., 0., -3., 6., -3., 3.,
                                               -10., 12., -3., -1., 0., -3., 14., -5., -9., 3., -3., 7.,
                                               -12., 13., 1., -3., 8., -1., 14., 13., -10., -8., 11., 11.,
                                               -2., 0., -8., -3., -2., 7., 8., -7., 5., -7., -9., 2.,
                                               -10., -4., 13., 7., 3., -4., -2., -4., 8., 0., -4., -12.,
                                               8., 8., -6., -15., -10., -11., -4., 10., -9., 8., 8., -5.,
                                               -5., -4., -9., -10., -3., -4., -15., -6., 0., -12., 7., -11.,
                                               7., 0., 11., 1., -6., 8., 9., 1., -15., 4., 1., 0.,
                                               5., 11., 14., -6., -14., 2., 10., 14., 5., 2., -12., 7.,
                                               -7., -2., -11., 12., 14., 12., 0., 6., -7., 5., -4., 0.,
                                               -15., -3., -6., 9., -12., -15., 3., 1., 13., -7., -13., 4.,
                                               -8., 3., 9., -9., -2., 3., 0., -3., -7., 7., 13., 6.,
                                               13., -14., -6., -9., 10., -9., -2., 4., -4., -15., -6., -1.,
                                               14., 6., 9., -1., 10., 6., -4., -11., -11., -6., 7., -7.,
                                               -15., -15., -7., 4., 3., 8., -7., -1., -15., -10., -5., -5.,
                                               -6., 11., -7., 1., -1., 11., 12., -10., 9., 2., -3., -2.,
                                               -4., 2., -10., -4., 7., 12., -11., -7., -7., -11., 11., 10.,
                                               -13., -2., 7., 4., 1., -8., -6., -3., 0., -8., 9., 13.,
                                               -6., 12., 4., -7., 4., -5., -3., -8., -9., -12., 3., -1.,
                                               -11., -9., -3., -9., 1., -6., 3., -3., 0., 10., -5., -9.,
                                               4., -7., -3., -11., -12., -6., -2., -5., 2., 2., 3., -1.,
                                               14., 12., 4., -6., -3., 9., 0., -6., -1., 9., 12., 4.,
                                               -1., 3., 14., -15., 6., -6., 2., 11., -7., -5., -14., 0.,
                                               -2., 1., -15., -13., 5., 0., -10., 10., 12., 13., -7., -6.,
                                               -15., -6., -1., 5., -4., 11., 12., 11., -11., -12., -13., 9.,
                                               9., -2., -7., -1., -2., 14., 9., 6., -3., 13., -7., -11.,
                                               -4., -14., -15., -5., -5., -5., -9., -2., -3., 1., 14., 8.,
                                               5., 11., 11., 7., -13., 13., -14., 8., -2., -12., -13., -10.,
                                               12., -8., -12., -1., 8., -7., 7., -6., -9., -4., 14., 8.,
                                               8., -15., 12., -14., -12., 10., -3., -2., 12., -3., 8., -3.,
                                               1., -5., 5., 14., 13., 11., -13., -15., -4., -10., -15., 0.,
                                               -3., -2., 14., -6., 4., 11., 1., -9., -14., -14., -8., 8.,
                                               4., -13., 12., -3., -2., -6., -14., 4., 8., 13., 10., -13.,
                                               -14., -5., 1., 2., -4., 5., -8., -4., 14., 6., -9., 6.,
                                               5., -15., 10., -3., 10., -13., 13., 10., -13., -8., -5., 5.,
                                               -15., 2., 6., -10., -14., 6., 5., -15., 11., -2., 11., 0.,
                                               2., -7., -5., -13., -9., -12., -9., -3., -8., 12., 9., 6.,
                                               -3., 8., 10., 12., -14., -5., -10., 2., 12., -2., 14., -15.,
                                               14., 13., 1., -6., 6., -9., 14., 12., 1., 3., -7., 8.,
                                               9., -8., 11., -14., -10., 1., 5., 6., 6., -4., -11., -11.,
                                               -10., -7., 10., 7., 10., -13., -6., -2., -5., 1., -14., 2.,
                                               -8., -2., -9., -1., -7., 10., -4., -6., 12., -1., -11., -8.,
                                               9., 9., 11., -2., 11., 13., 2., 4., 8., -5., 1., -2.,
                                               -13., -10., 4., -12., -14., 7., -14., 9., -11., 1., -5., 2.,
                                               -8., 0., -11., -8., 8., 11., -12., -8., 4., -9., -9., -3.,
                                               3., 14., 5., -6., 5., 1., 7., 11., -14., -12., 7., -14.,
                                               12., -8., -11., -11., -9., -5., -7., -5., -9., -13., -14., -9.,
                                               14., 11., -11., -11., -4., -9., -13., -1., 8., -14., -5., 7.,
                                               -9., 14., 6., -8., -13., 8., 1., -8., 0., 7., -14., -5.,
                                               4., -3., 10., -13., -14., -11., 0., -8., -15., -8., 12., -12.,
                                               5., -5., -8., 2., -15., -11., -11., 6., -6., 1., -2., 1.,
                                               6., -6., -1., -3., 14., -7., 1., -10., -9., -8., 13., 14.,
                                               -3., 13., 14., -3., 8., -3., 11., 11., -6., -9., 11., 7.,
                                               -4., 1., -12., -12., -8., -4., -2., -3., 9., 14., -11., 13.,
                                               12., -6., -6., -12., -2., -10., 2., 3., -2., -6., 13., 12.,
                                               4., 11., -1., 0., -4., 8., -11., -3., 12., 4., 10., -2.,
                                               7., -10., 13., -9., 6., 9., -14., -8., -13., 5., 14., 5.,
                                               13., -12., -7., 4., 7., -1., 5., -7., 10., 7., -9., 2.,
                                               1., -6., 10., 10., -5., -8., 12., 12., 8., 8., -6., -11.,
                                               14., -3., 13., 2., 13., 11., -8., -11., -14., 12., -3., 9.,
                                               -2., 14., 8., 12., 8., 14., 5., 0., 11., -10., -7., 12.,
                                               -14., 1., -1., 13., 7., -2., 10., 10., -12., 3., 10., -3.,
                                               6., 11., 2., -12., -9., -10., 8., 8., 14., -4., -1., -7.,
                                               2., 2., 9., -3., -9., 3., 10., -10., 3., 2., -4., 2.,
                                               3., 5., -11., 13., -11., -15., -10., 13., 13., 8., -5., -11.,
                                               -9., -2., -9., -5., 14., -13., 5., 13., -7., -10., 1., -9.,
                                               11., 4., -11., 0., 0., -11., 14., -15., -15., 11., -10., 13.,
                                               1., -13., 13., 6., -1., -7., 0., 6., -14., 1., 14., -4.,
                                               5., -11., 8., 9., -1., -9., 1., 9., 11., -10., 6., -9.,
                                               12., 2., -3., 5., -15., 1., -4., 3., 6., 6., 13., 5.,
                                               -10., -2., 2., 8., 1., 3., 12., 6., 0., 8., 11., -6.,
                                               -13., -3., -14., 11., -13., 10., -2., 3., -6., 10., -15., 14.});
    DTensor
      value_v2 = tensor::Tensor<double>({6, 4, 2, 2}, {0., 14., -13., -1., 1., -4., 0., 6., 8., -11., 11., 11.,
                                                       -6., 11., -14., -7., 9., 7., 13., -13., 11., 2., 8., 8.,
                                                       -3., 12., 8., 4., -10., -1., 0., 7., -7., -12., -13., -13.,
                                                       13., 1., 1., -10., -14., 8., 11., 4., -1., -14., 6., 10.,
                                                       -8., 13., 4., -7., -8., 12., 9., 12., -9., 6., 7., 8.,
                                                       -8., -14., 8., -12., 11., -7., -7., -12., -11., -6., -12., -3.,
                                                       -14., 5., 14., -11., -8., 14., -13., 0., 1., -12., 1., -5.,
                                                       -13., -6., -6., -2., -11., -5., 13., -1., 8., 8., 1., -15.});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto conv2d = functional::Conv2D(&v1, &v2, {2, 2});
    auto target = functional::ReduceSum(&conv2d);

    graph->zero_grad();
    target.forward();

    // test forward
    std::vector<double> conv2d_exp = {-314., -507., -685., -274., 23., 35., 401., -377., 79., 299.,
                                      358., -516., 68., -477., -403., -33., 30., -29., -16., 25.,
                                      -844., 12., -121., -395., 57., -65., -293., 500., 255., 179.,
                                      -169., -38., -22., 374., -16., -79., -147., -424., 392., -20.,
                                      2., 290., 753., -115., -442., 380., 624., 683., -392., 68.,
                                      -277., 727., -366., 359., -125., 673., -403., -538., 306., 536.,
                                      10., 293., 10., -257., 273., -88., -139., 635., -265., 378.,
                                      220., 105., -48., 196., -399., -131., -30., -255., 723., 137.,
                                      -98., 173., -479., 5., 172., 112., -141., 317., 142., 540.,
                                      85., -216., -143., 155., -241., 309., -393., -193., -478., 246.,
                                      172., -48., -151., 102., 215., 1., 103., 200., -426., 385.,
                                      137., -83., 13., -31., -193., 65., -273., 65., -342., 354.,
                                      -235., -75., 53., -304., -263., 0., -432., 307., -183., 433.,
                                      -112., -83., 257., 438., 370., 126., 488., 141., -441., 748.,
                                      -352., -249., -414., -129., 375., -47., 53., -319., -287., 124.,
                                      -408., -269., -264., 208., -324., 651., 118., -566., 62., -107.,
                                      175., -446., 278., 312., 129., 488., 329., -436., -482., -749.,
                                      37., -171., -49., -284., -65., -240., 203., 325., 254., -303.,
                                      -292., -436., 178., 155., 45., 54., -254., 128., 87., -179.,
                                      -179., -604., -595., -495., 147., 253., 446., -177., 77., 146.,
                                      264., -374., 103., 213., -9., -220., -184., 128., -433., 134.,
                                      571., -162., 69., -311., -77., 167., 327., 411., 560., -32.,
                                      306., -368., 381., 465., -259., 18., -544., 177., -292., 2.,
                                      359., 15., 223., -501., 30., -276., 441., 293., 93., 240.,
                                      -151., 549., -72., 119., 321., -92., -60., 84., 111., -191.,
                                      -286., 119., -401., -57., 198., 116., 133., -30., -346., -133.,
                                      26., -106., -437., 165., 45., 377., -814., 450., -355., 365.,
                                      156., -394., 171., 194., 31., -72., 223., 311., -12., 307.,
                                      76., -241., 471., 604., 297., 113., -106., 358., -134., 564.,
                                      179., 349., 455., -131., 288., 316., 61., -48., 644., -272.,
                                      183., -190., 6., -400., -586., 555., -63., 141., 146., -123.,
                                      217., 251., -229., -520., -201., -586., 111., -7., -290., -152.,
                                      -155., -125., -469., 22., 498., -759., -99., -583., -30., 856.,
                                      -228., -212., 159., 262., -103., -113., 199., -9., 222., -218.,
                                      -306., -15., 74., -359., 98., -583., 491., -214., 90., 353.,
                                      -268., 132., 351., -66., -249., -249., 215., -227., -135., -516.};
    ASSERT_EQ(conv2d.get_value_shape(), tensor::TensorShape({3, 6, 4, 5}));
    std::vector<double> conv2d_out = conv2d.get_value().to_vector();
    ASSERT_THAT(conv2d_out,
                ElementsAreArray(conv2d_exp)) << "\nResult: \n" << conv2d.get_value().to_string();

    ASSERT_EQ(target.get_value().get_value(), 320);
//
//    Variable v3 = Variable({100, 3, 200, 200});
//    auto conv2d__ = functional::Conv2D(&v3, &v2, {1, 1});
//    SUCCEED() << conv2d__.get_value().to_string() << std::endl;

    graph->backward(target);

    std::vector<double> conv2d_grad_exp =
      {6., 3., 6., 3., 6., 3., 6., 3., 6., 3., -15., -51.,
       -15., -51., -15., -51., -15., -51., -15., -51., 6., 3., 6., 3.,
       6., 3., 6., 3., 6., 3., -15., -51., -15., -51., -15., -51.,
       -15., -51., -15., -51., 6., 3., 6., 3., 6., 3., 6., 3.,
       6., 3., -15., -51., -15., -51., -15., -51., -15., -51., -15., -51.,
       6., 3., 6., 3., 6., 3., 6., 3., 6., 3., -15., -51.,
       -15., -51., -15., -51., -15., -51., -15., -51., -7., -1., -7., -1.,
       -7., -1., -7., -1., -7., -1., 0., 11., 0., 11., 0., 11.,
       0., 11., 0., 11., -7., -1., -7., -1., -7., -1., -7., -1.,
       -7., -1., 0., 11., 0., 11., 0., 11., 0., 11., 0., 11.,
       -7., -1., -7., -1., -7., -1., -7., -1., -7., -1., 0., 11.,
       0., 11., 0., 11., 0., 11., 0., 11., -7., -1., -7., -1.,
       -7., -1., -7., -1., -7., -1., 0., 11., 0., 11., 0., 11.,
       0., 11., 0., 11., -43., 15., -43., 15., -43., 15., -43., 15.,
       -43., 15., 64., 15., 64., 15., 64., 15., 64., 15., 64., 15.,
       -43., 15., -43., 15., -43., 15., -43., 15., -43., 15., 64., 15.,
       64., 15., 64., 15., 64., 15., 64., 15., -43., 15., -43., 15.,
       -43., 15., -43., 15., -43., 15., 64., 15., 64., 15., 64., 15.,
       64., 15., 64., 15., -43., 15., -43., 15., -43., 15., -43., 15.,
       -43., 15., 64., 15., 64., 15., 64., 15., 64., 15., 64., 15.,
       -25., 4., -25., 4., -25., 4., -25., 4., -25., 4., -12., -17.,
       -12., -17., -12., -17., -12., -17., -12., -17., -25., 4., -25., 4.,
       -25., 4., -25., 4., -25., 4., -12., -17., -12., -17., -12., -17.,
       -12., -17., -12., -17., -25., 4., -25., 4., -25., 4., -25., 4.,
       -25., 4., -12., -17., -12., -17., -12., -17., -12., -17., -12., -17.,
       -25., 4., -25., 4., -25., 4., -25., 4., -25., 4., -12., -17.,
       -12., -17., -12., -17., -12., -17., -12., -17., 6., 3., 6., 3.,
       6., 3., 6., 3., 6., 3., -15., -51., -15., -51., -15., -51.,
       -15., -51., -15., -51., 6., 3., 6., 3., 6., 3., 6., 3.,
       6., 3., -15., -51., -15., -51., -15., -51., -15., -51., -15., -51.,
       6., 3., 6., 3., 6., 3., 6., 3., 6., 3., -15., -51.,
       -15., -51., -15., -51., -15., -51., -15., -51., 6., 3., 6., 3.,
       6., 3., 6., 3., 6., 3., -15., -51., -15., -51., -15., -51.,
       -15., -51., -15., -51., -7., -1., -7., -1., -7., -1., -7., -1.,
       -7., -1., 0., 11., 0., 11., 0., 11., 0., 11., 0., 11.,
       -7., -1., -7., -1., -7., -1., -7., -1., -7., -1., 0., 11.,
       0., 11., 0., 11., 0., 11., 0., 11., -7., -1., -7., -1.,
       -7., -1., -7., -1., -7., -1., 0., 11., 0., 11., 0., 11.,
       0., 11., 0., 11., -7., -1., -7., -1., -7., -1., -7., -1.,
       -7., -1., 0., 11., 0., 11., 0., 11., 0., 11., 0., 11.,
       -43., 15., -43., 15., -43., 15., -43., 15., -43., 15., 64., 15.,
       64., 15., 64., 15., 64., 15., 64., 15., -43., 15., -43., 15.,
       -43., 15., -43., 15., -43., 15., 64., 15., 64., 15., 64., 15.,
       64., 15., 64., 15., -43., 15., -43., 15., -43., 15., -43., 15.,
       -43., 15., 64., 15., 64., 15., 64., 15., 64., 15., 64., 15.,
       -43., 15., -43., 15., -43., 15., -43., 15., -43., 15., 64., 15.,
       64., 15., 64., 15., 64., 15., 64., 15., -25., 4., -25., 4.,
       -25., 4., -25., 4., -25., 4., -12., -17., -12., -17., -12., -17.,
       -12., -17., -12., -17., -25., 4., -25., 4., -25., 4., -25., 4.,
       -25., 4., -12., -17., -12., -17., -12., -17., -12., -17., -12., -17.,
       -25., 4., -25., 4., -25., 4., -25., 4., -25., 4., -12., -17.,
       -12., -17., -12., -17., -12., -17., -12., -17., -25., 4., -25., 4.,
       -25., 4., -25., 4., -25., 4., -12., -17., -12., -17., -12., -17.,
       -12., -17., -12., -17., 6., 3., 6., 3., 6., 3., 6., 3.,
       6., 3., -15., -51., -15., -51., -15., -51., -15., -51., -15., -51.,
       6., 3., 6., 3., 6., 3., 6., 3., 6., 3., -15., -51.,
       -15., -51., -15., -51., -15., -51., -15., -51., 6., 3., 6., 3.,
       6., 3., 6., 3., 6., 3., -15., -51., -15., -51., -15., -51.,
       -15., -51., -15., -51., 6., 3., 6., 3., 6., 3., 6., 3.,
       6., 3., -15., -51., -15., -51., -15., -51., -15., -51., -15., -51.,
       -7., -1., -7., -1., -7., -1., -7., -1., -7., -1., 0., 11.,
       0., 11., 0., 11., 0., 11., 0., 11., -7., -1., -7., -1.,
       -7., -1., -7., -1., -7., -1., 0., 11., 0., 11., 0., 11.,
       0., 11., 0., 11., -7., -1., -7., -1., -7., -1., -7., -1.,
       -7., -1., 0., 11., 0., 11., 0., 11., 0., 11., 0., 11.,
       -7., -1., -7., -1., -7., -1., -7., -1., -7., -1., 0., 11.,
       0., 11., 0., 11., 0., 11., 0., 11., -43., 15., -43., 15.,
       -43., 15., -43., 15., -43., 15., 64., 15., 64., 15., 64., 15.,
       64., 15., 64., 15., -43., 15., -43., 15., -43., 15., -43., 15.,
       -43., 15., 64., 15., 64., 15., 64., 15., 64., 15., 64., 15.,
       -43., 15., -43., 15., -43., 15., -43., 15., -43., 15., 64., 15.,
       64., 15., 64., 15., 64., 15., 64., 15., -43., 15., -43., 15.,
       -43., 15., -43., 15., -43., 15., 64., 15., 64., 15., 64., 15.,
       64., 15., 64., 15., -25., 4., -25., 4., -25., 4., -25., 4.,
       -25., 4., -12., -17., -12., -17., -12., -17., -12., -17., -12., -17.,
       -25., 4., -25., 4., -25., 4., -25., 4., -25., 4., -12., -17.,
       -12., -17., -12., -17., -12., -17., -12., -17., -25., 4., -25., 4.,
       -25., 4., -25., 4., -25., 4., -12., -17., -12., -17., -12., -17.,
       -12., -17., -12., -17., -25., 4., -25., 4., -25., 4., -25., 4.,
       -25., 4., -12., -17., -12., -17., -12., -17., -12., -17., -12., -17.};
    std::vector<double> kernel_grad_exp = {-118., -152., -91., -101., 11., 15., 12., -62., 52., 26.,
                                           -39., 13., 30., -97., -121., 25., -118., -152., -91., -101.,
                                           11., 15., 12., -62., 52., 26., -39., 13., 30., -97.,
                                           -121., 25., -118., -152., -91., -101., 11., 15., 12., -62.,
                                           52., 26., -39., 13., 30., -97., -121., 25., -118., -152.,
                                           -91., -101., 11., 15., 12., -62., 52., 26., -39., 13.,
                                           30., -97., -121., 25., -118., -152., -91., -101., 11., 15.,
                                           12., -62., 52., 26., -39., 13., 30., -97., -121., 25.,
                                           -118., -152., -91., -101., 11., 15., 12., -62., 52., 26.,
                                           -39., 13., 30., -97., -121., 25.};
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAreArray(kernel_grad_exp))
              << "\nResult:\n" << v2.get_grad().to_string();
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAreArray(conv2d_grad_exp))
              << "\nResult:\n" << v1.get_grad().to_string();

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}