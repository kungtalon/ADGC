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
    Variable v1 = Variable({4, 10, 10, 3}); // [B, H, W, C]
    Parameter v2 = Parameter({5, 3, 2, 2}); // [c_out, c_in, kh, kw]

    DTensor value_v1 = tensor::Tensor<double>({4, 10, 10, 3},
                                              {11.0, 2.0, 9.0, 0.0, 9.0, 9.0, 6.0, 17.0, 2.0, 3.0, 7.0, 14.0, 2.0, 3.0,
                                               1.0, 3.0, 3.0, 8.0, 1.0, 6.0, 2.0, 18.0, 13.0, 15.0, 9.0, 6.0, 19.0, 3.0,
                                               11.0, 16.0, 15.0, 19.0, 1.0, 18.0, 19.0, 7.0, 18.0, 17.0, 3.0, 8.0, 9.0,
                                               9.0, 8.0, 15.0, 6.0, 15.0, 14.0, 7.0, 8.0, 4.0, 8.0, 6.0, 12.0, 6.0,
                                               19.0, 9.0, 8.0, 5.0, 7.0, 6.0, 3.0, 14.0, 9.0, 11.0, 14.0, 1.0, 14.0,
                                               11.0, 2.0, 3.0, 11.0, 1.0, 19.0, 10.0, 4.0, 4.0, 10.0, 1.0, 3.0, 8.0,
                                               19.0, 19.0, 19.0, 13.0, 7.0, 0.0, 9.0, 18.0, 9.0, 9.0, 5.0, 5.0, 2.0,
                                               1.0, 4.0, 3.0, 3.0, 17.0, 1.0, 10.0, 7.0, 19.0, 4.0, 12.0, 11.0, 9.0,
                                               17.0, 6.0, 17.0, 16.0, 18.0, 12.0, 11.0, 8.0, 5.0, 14.0, 1.0, 3.0, 19.0,
                                               9.0, 9.0, 13.0, 7.0, 8.0, 4.0, 15.0, 12.0, 4.0, 16.0, 4.0, 11.0, 19.0,
                                               19.0, 0.0, 3.0, 11.0, 3.0, 16.0, 13.0, 10.0, 5.0, 14.0, 3.0, 3.0, 14.0,
                                               6.0, 16.0, 19.0, 18.0, 18.0, 8.0, 4.0, 0.0, 8.0, 16.0, 5.0, 13.0, 2.0,
                                               15.0, 1.0, 15.0, 13.0, 12.0, 19.0, 0.0, 2.0, 18.0, 7.0, 5.0, 12.0, 0.0,
                                               2.0, 9.0, 3.0, 17.0, 9.0, 18.0, 4.0, 15.0, 14.0, 18.0, 3.0, 3.0, 0.0,
                                               19.0, 2.0, 0.0, 14.0, 10.0, 18.0, 3.0, 6.0, 13.0, 6.0, 13.0, 18.0, 3.0,
                                               7.0, 15.0, 7.0, 11.0, 13.0, 6.0, 19.0, 12.0, 5.0, 8.0, 19.0, 2.0, 16.0,
                                               7.0, 11.0, 14.0, 5.0, 0.0, 15.0, 7.0, 3.0, 13.0, 4.0, 14.0, 3.0, 12.0,
                                               12.0, 9.0, 12.0, 12.0, 0.0, 16.0, 2.0, 12.0, 6.0, 16.0, 17.0, 8.0, 14.0,
                                               19.0, 8.0, 7.0, 10.0, 15.0, 15.0, 15.0, 6.0, 15.0, 5.0, 13.0, 7.0, 2.0,
                                               2.0, 7.0, 7.0, 10.0, 2.0, 16.0, 5.0, 12.0, 9.0, 6.0, 6.0, 13.0, 5.0, 5.0,
                                               2.0, 5.0, 15.0, 4.0, 5.0, 17.0, 1.0, 17.0, 10.0, 10.0, 18.0, 1.0, 4.0,
                                               16.0, 16.0, 11.0, 16.0, 13.0, 15.0, 2.0, 9.0, 3.0, 16.0, 6.0, 10.0, 11.0,
                                               19.0, 11.0, 18.0, 2.0, 1.0, 12.0, 13.0, 19.0, 12.0, 15.0, 11.0, 3.0, 1.0,
                                               0.0, 2.0, 13.0, 5.0, 16.0, 17.0, 2.0, 4.0, 11.0, 12.0, 14.0, 6.0, 5.0,
                                               13.0, 19.0, 16.0, 15.0, 11.0, 0.0, 3.0, 15.0, 14.0, 4.0, 3.0, 2.0, 9.0,
                                               0.0, 15.0, 14.0, 15.0, 9.0, 5.0, 7.0, 14.0, 2.0, 4.0, 4.0, 18.0, 13.0,
                                               5.0, 6.0, 17.0, 16.0, 15.0, 13.0, 19.0, 4.0, 6.0, 19.0, 0.0, 17.0, 1.0,
                                               6.0, 6.0, 3.0, 5.0, 11.0, 4.0, 8.0, 7.0, 5.0, 13.0, 17.0, 19.0, 8.0,
                                               18.0, 19.0, 8.0, 14.0, 14.0, 12.0, 14.0, 2.0, 18.0, 5.0, 11.0, 15.0, 1.0,
                                               15.0, 0.0, 17.0, 16.0, 15.0, 15.0, 19.0, 3.0, 0.0, 13.0, 5.0, 7.0, 7.0,
                                               1.0, 7.0, 12.0, 8.0, 18.0, 19.0, 13.0, 12.0, 13.0, 0.0, 19.0, 10.0, 14.0,
                                               2.0, 11.0, 11.0, 5.0, 18.0, 1.0, 3.0, 9.0, 2.0, 11.0, 7.0, 1.0, 15.0,
                                               11.0, 7.0, 13.0, 13.0, 15.0, 13.0, 1.0, 6.0, 15.0, 10.0, 14.0, 15.0, 3.0,
                                               10.0, 0.0, 4.0, 1.0, 17.0, 2.0, 3.0, 10.0, 8.0, 16.0, 16.0, 6.0, 12.0,
                                               8.0, 12.0, 8.0, 14.0, 6.0, 18.0, 17.0, 8.0, 0.0, 13.0, 13.0, 2.0, 11.0,
                                               18.0, 8.0, 6.0, 0.0, 14.0, 10.0, 17.0, 14.0, 15.0, 9.0, 0.0, 19.0, 1.0,
                                               4.0, 15.0, 18.0, 13.0, 10.0, 3.0, 10.0, 5.0, 12.0, 9.0, 19.0, 6.0, 16.0,
                                               16.0, 5.0, 12.0, 0.0, 16.0, 15.0, 15.0, 12.0, 13.0, 17.0, 17.0, 17.0,
                                               4.0, 18.0, 13.0, 2.0, 13.0, 19.0, 2.0, 7.0, 19.0, 18.0, 9.0, 11.0, 1.0,
                                               0.0, 6.0, 8.0, 15.0, 2.0, 4.0, 16.0, 18.0, 9.0, 8.0, 5.0, 13.0, 11.0,
                                               14.0, 3.0, 0.0, 19.0, 0.0, 11.0, 13.0, 12.0, 6.0, 3.0, 8.0, 8.0, 2.0,
                                               12.0, 0.0, 4.0, 15.0, 9.0, 19.0, 10.0, 5.0, 8.0, 9.0, 18.0, 13.0, 3.0,
                                               17.0, 8.0, 14.0, 8.0, 1.0, 18.0, 9.0, 6.0, 9.0, 4.0, 5.0, 18.0, 17.0,
                                               11.0, 13.0, 18.0, 10.0, 18.0, 6.0, 8.0, 12.0, 4.0, 13.0, 6.0, 11.0, 14.0,
                                               6.0, 18.0, 1.0, 10.0, 0.0, 0.0, 10.0, 17.0, 9.0, 12.0, 3.0, 6.0, 5.0,
                                               17.0, 10.0, 18.0, 8.0, 1.0, 11.0, 9.0, 5.0, 13.0, 1.0, 9.0, 0.0, 0.0,
                                               8.0, 4.0, 9.0, 1.0, 15.0, 14.0, 15.0, 9.0, 12.0, 12.0, 8.0, 5.0, 14.0,
                                               18.0, 2.0, 8.0, 15.0, 19.0, 10.0, 3.0, 4.0, 11.0, 11.0, 13.0, 16.0, 14.0,
                                               2.0, 3.0, 12.0, 7.0, 3.0, 7.0, 16.0, 0.0, 17.0, 17.0, 16.0, 3.0, 13.0,
                                               17.0, 18.0, 6.0, 16.0, 2.0, 15.0, 0.0, 0.0, 7.0, 8.0, 3.0, 4.0, 15.0,
                                               13.0, 18.0, 9.0, 10.0, 7.0, 3.0, 12.0, 15.0, 14.0, 14.0, 19.0, 6.0, 8.0,
                                               6.0, 8.0, 2.0, 15.0, 7.0, 17.0, 2.0, 11.0, 11.0, 3.0, 18.0, 19.0, 14.0,
                                               9.0, 12.0, 11.0, 0.0, 3.0, 5.0, 1.0, 14.0, 7.0, 9.0, 5.0, 6.0, 1.0, 19.0,
                                               19.0, 19.0, 6.0, 2.0, 15.0, 2.0, 14.0, 7.0, 16.0, 12.0, 14.0, 13.0, 2.0,
                                               12.0, 11.0, 4.0, 8.0, 17.0, 16.0, 13.0, 9.0, 12.0, 0.0, 15.0, 18.0, 15.0,
                                               14.0, 13.0, 16.0, 0.0, 17.0, 13.0, 13.0, 13.0, 5.0, 13.0, 4.0, 11.0, 3.0,
                                               4.0, 0.0, 13.0, 14.0, 3.0, 19.0, 13.0, 11.0, 16.0, 9.0, 11.0, 2.0, 0.0,
                                               0.0, 4.0, 8.0, 15.0, 19.0, 14.0, 14.0, 16.0, 6.0, 18.0, 19.0, 11.0, 11.0,
                                               12.0, 14.0, 9.0, 16.0, 17.0, 18.0, 9.0, 16.0, 0.0, 14.0, 6.0, 11.0, 7.0,
                                               17.0, 8.0, 19.0, 3.0, 10.0, 4.0, 15.0, 13.0, 11.0, 8.0, 15.0, 19.0, 6.0,
                                               15.0, 13.0, 2.0, 2.0, 4.0, 15.0, 8.0, 8.0, 7.0, 14.0, 14.0, 12.0, 6.0,
                                               6.0, 19.0, 12.0, 0.0, 11.0, 7.0, 8.0, 2.0, 18.0, 17.0, 19.0, 13.0, 16.0,
                                               12.0, 7.0, 19.0, 13.0, 4.0, 5.0, 8.0, 9.0, 14.0, 16.0, 4.0, 2.0, 7.0,
                                               9.0, 15.0, 16.0, 19.0, 12.0, 4.0, 2.0, 9.0, 19.0, 12.0, 3.0, 0.0, 4.0,
                                               8.0, 18.0, 2.0, 6.0, 17.0, 17.0, 9.0, 13.0, 0.0, 3.0, 3.0, 7.0, 7.0,
                                               10.0, 0.0, 1.0, 13.0, 3.0, 7.0, 18.0, 19.0, 13.0, 18.0, 1.0, 12.0, 11.0,
                                               11.0, 9.0, 15.0, 0.0, 4.0, 15.0, 6.0, 16.0, 19.0, 13.0, 1.0, 13.0, 7.0,
                                               16.0, 5.0, 15.0, 2.0, 8.0, 19.0, 19.0, 5.0, 16.0, 13.0, 10.0, 14.0, 11.0,
                                               2.0, 8.0, 19.0, 8.0, 4.0, 15.0, 5.0, 10.0, 15.0, 7.0, 19.0, 19.0, 15.0,
                                               0.0, 1.0, 9.0, 12.0, 7.0, 5.0, 1.0, 4.0, 16.0, 10.0, 2.0, 1.0, 4.0, 9.0,
                                               5.0, 10.0, 9.0, 4.0, 3.0, 11.0, 19.0, 2.0, 17.0, 3.0, 14.0, 3.0, 4.0,
                                               13.0, 17.0, 8.0, 7.0, 3.0, 5.0, 14.0, 8.0, 6.0, 1.0, 7.0, 18.0, 1.0,
                                               12.0, 13.0, 5.0, 17.0, 18.0, 12.0, 3.0, 3.0, 15.0, 19.0, 10.0, 6.0, 2.0,
                                               16.0, 7.0, 7.0, 18.0, 18.0, 10.0, 12.0, 3.0, 16.0, 5.0, 4.0, 11.0, 5.0,
                                               7.0, 3.0, 8.0, 6.0, 14.0, 0.0, 11.0, 17.0, 2.0, 10.0, 10.0, 17.0, 5.0,
                                               15.0, 8.0, 16.0, 3.0, 12.0, 14.0, 15.0, 18.0, 14.0, 5.0, 12.0, 17.0,
                                               17.0, 5.0, 8.0, 15.0, 11.0, 2.0, 11.0, 17.0, 10.0, 12.0, 3.0, 5.0, 15.0,
                                               8.0, 16.0, 17.0, 0.0, 17.0, 5.0, 13.0, 4.0, 11.0, 6.0, 15.0, 0.0, 10.0,
                                               14.0, 19.0, 8.0, 8.0, 2.0, 16.0, 15.0, 15.0, 14.0, 5.0, 18.0, 8.0, 11.0,
                                               16.0, 13.0, 12.0, 17.0, 2.0, 18.0, 2.0, 3.0, 13.0, 8.0, 12.0, 5.0, 1.0,
                                               4.0, 4.0, 0.0, 9.0, 15.0, 8.0, 12.0, 9.0, 7.0, 12.0, 5.0, 14.0, 0.0,
                                               16.0, 19.0, 8.0, 10.0, 8.0, 17.0, 18.0, 12.0, 10.0, 6.0, 0.0, 4.0, 6.0,
                                               0.0, 13.0, 15.0, 9.0, 10.0, 13.0, 17.0, 12.0, 13.0, 5.0, 0.0, 7.0, 10.0,
                                               10.0, 10.0, 2.0, 13.0, 8.0, 9.0, 17.0, 16.0, 2.0, 12.0, 18.0, 6.0, 5.0,
                                               0.0, 18.0, 17.0, 16.0, 12.0, 1.0, 0.0, 7.0, 1.0, 0.0, 5.0, 11.0, 1.0,
                                               9.0, 11.0, 17.0, 16.0, 6.0, 16.0, 9.0, 5.0, 8.0, 4.0, 4.0, 0.0, 9.0,
                                               12.0, 14.0, 16.0, 17.0, 0.0, 12.0, 10.0, 16.0, 12.0, 10.0, 3.0, 17.0,
                                               9.0, 9.0, 2.0, 6.0, 13.0, 3.0, 11.0, 9.0, 16.0, 1.0, 7.0, 16.0, 7.0, 8.0,
                                               10.0, 19.0, 14.0, 0.0, 19.0, 5.0, 8.0, 12.0, 17.0, 3.0, 19.0, 16.0, 2.0,
                                               6.0, 1.0, 0.0, 3.0, 2.0, 5.0, 1.0, 4.0, 19.0, 2.0, 8.0, 4.0, 1.0, 10.0,
                                               9.0, 9.0, 2.0, 2.0, 12.0, 0.0, 19.0, 14.0, 17.0, 3.0, 17.0, 14.0, 10.0,
                                               10.0, 5.0, 15.0, 9.0, 12.0, 13.0, 9.0, 8.0, 6.0, 12.0, 18.0, 8.0, 5.0});
    DTensor
      value_v2 = tensor::Tensor<double>({5, 3, 2, 2}, {10., 12., 16., 4., 3., 8., 12., 6., 4., 14., 19., 7., 15., 5.,
                                                       6., 12., 5., 8., 15., 12., 12., 16., 13., 4., 10., 3., 16., 14.,
                                                       4., 17., 17., 12., 8., 18., 0., 10., 17., 11., 2., 12., 6., 0.,
                                                       13., 2., 17., 15., 6., 4., 3., 13., 13., 10., 15., 16., 2., 9.,
                                                       12., 8., 0., 17.});

    v1.assign_value(value_v1);
    v2.assign_value(value_v2);

    auto conv2d = functional::Conv2D(&v1, &v2, {2, 2});
    auto target = functional::ReduceSum(&conv2d);

    graph->zero_grad();
    target.forward();

    tensor::Tensor<double> col_img = conv2d.get_im2col();
//    FAIL() << col_img.to_string();

    // test forward
    std::vector<double> conv2d_exp = {1072., 1359., 1618., 1052., 1090., 1105., 1136., 1411., 906., 1142.,
                                      820., 951., 1075., 705., 743., 1032., 955., 1079., 746., 1061.,
                                      1200., 1273., 1421., 1086., 1168., 593., 609., 704., 543., 874.,
                                      805., 974., 1165., 829., 1013., 975., 1182., 1120., 906., 896.,
                                      1668., 1652., 1752., 1374., 1620., 976., 1072., 1189., 913., 1026.,
                                      834., 1012., 1110., 879., 1015., 1300., 1351., 1371., 1065., 1207.,
                                      1167., 1281., 1415., 1094., 907., 721., 801., 807., 785., 744.,
                                      1772., 1714., 1780., 1418., 1655., 1016., 1014., 1094., 766., 873.,
                                      922., 884., 766., 761., 974., 1165., 1264., 1197., 1183., 1031.,
                                      1432., 1423., 1485., 1214., 1404., 1436., 1393., 1245., 1281., 1059.,
                                      1205., 1336., 1439., 1199., 1306., 1241., 1319., 1483., 1008., 1134.,
                                      819., 1107., 1198., 1027., 1009., 962., 1111., 1109., 1003., 821.,
                                      1245., 1257., 1335., 838., 1257., 966., 870., 1124., 561., 861.,
                                      864., 1165., 1217., 974., 1067., 1585., 1684., 1795., 1353., 1495.,
                                      1141., 1208., 1090., 802., 835., 714., 757., 805., 650., 625.,
                                      1106., 1090., 1235., 908., 1138., 1534., 1731., 1708., 1461., 1561.,
                                      1143., 1231., 1182., 1208., 1055., 1236., 1261., 1252., 1005., 983.,
                                      957., 1255., 1128., 1171., 1021., 1272., 1196., 1373., 981., 1389.,
                                      1049., 1102., 954., 924., 991., 1168., 1210., 1320., 773., 1009.,
                                      975., 1093., 1301., 916., 1286., 1023., 1107., 1316., 986., 1214.,
                                      1182., 1077., 1092., 922., 1310., 1375., 1366., 1541., 1152., 1365.,
                                      1150., 1290., 1428., 1408., 1492., 1112., 1084., 1089., 1070., 991.,
                                      765., 1002., 946., 979., 972., 1367., 1418., 1529., 1248., 1340.,
                                      842., 1038., 1160., 956., 1092., 818., 852., 1059., 746., 1170.,
                                      1321., 1430., 1538., 1245., 1412., 717., 858., 947., 821., 881.,
                                      1113., 1200., 1219., 829., 1052., 1260., 1372., 1365., 1133., 1296.,
                                      1298., 1137., 1209., 929., 1163., 1031., 1117., 1323., 929., 1229.,
                                      1026., 1092., 1221., 838., 1099., 1209., 1544., 1307., 1412., 1063.,
                                      1132., 1139., 1238., 814., 1230., 1279., 1319., 1592., 1103., 1379.,
                                      1090., 1254., 1194., 1042., 980., 1069., 1209., 1105., 972., 1069.,
                                      1463., 1711., 1876., 1469., 1485., 1154., 1127., 1158., 882., 1171.,
                                      1291., 1251., 1399., 845., 1290., 1092., 1152., 978., 918., 1015.,
                                      1239., 1202., 1322., 907., 1270., 1232., 1239., 1207., 1095., 1171.,
                                      1043., 1018., 974., 882., 856., 1420., 1681., 1603., 1454., 1235.,
                                      907., 1017., 790., 883., 594., 1424., 1510., 1397., 1378., 1489.,
                                      941., 1086., 980., 862., 884., 962., 1176., 1247., 916., 1011.,
                                      1445., 1512., 1629., 1337., 1436., 1153., 1259., 1339., 1031., 1124.,
                                      1306., 1554., 1827., 1213., 1502., 698., 749., 836., 657., 842.,
                                      845., 862., 659., 727., 791., 1048., 1062., 974., 878., 810.,
                                      1147., 1187., 1406., 984., 1307., 1323., 1367., 1379., 1084., 1219.,
                                      1074., 1115., 1386., 806., 1234., 915., 1032., 935., 744., 771.,
                                      1185., 1431., 1463., 1167., 1146., 1214., 1092., 1094., 938., 1160.,
                                      1289., 1451., 1431., 1296., 1483., 1206., 1284., 1269., 1064., 1100.,
                                      1421., 1421., 1710., 1176., 1591., 1127., 949., 962., 923., 991.,
                                      853., 1009., 876., 916., 895., 1210., 1204., 1248., 1051., 917.,
                                      1189., 1289., 1414., 1162., 1136., 941., 973., 974., 993., 1096.,
                                      1481., 1505., 1596., 1084., 1120., 1037., 1066., 1149., 730., 928.,
                                      726., 808., 968., 679., 917., 804., 887., 1115., 696., 1128.,
                                      1116., 1254., 1090., 1154., 1175., 1433., 1368., 1361., 1212., 1327.,
                                      864., 1083., 1022., 907., 672., 850., 897., 1047., 573., 918.};
    ASSERT_EQ(conv2d.get_value_shape(), tensor::TensorShape({4, 5, 5, 5}));
    std::vector<double> conv2d_out = conv2d.get_value().to_vector();
    ASSERT_THAT(conv2d_out,
                ElementsAreArray(conv2d_exp)) << "\nResult: \n" << conv2d.get_value().to_string();

    ASSERT_EQ(target.get_value().get_value(), 565654);

    graph->backward(target);

    std::vector<double> conv2d_grad_exp =
      {55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0,
       49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0,
       42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0,
       53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0,
       49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0,
       71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0,
       53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0,
       53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0,
       49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0,
       71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0,
       53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0,
       53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0,
       41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0,
       71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0,
       53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0,
       59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0,
       53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0,
       41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0,
       71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0,
       53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0,
       59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0,
       53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0,
       44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0,
       41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0,
       71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0,
       55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0,
       59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0,
       53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0,
       44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0,
       41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0,
       42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0,
       55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0,
       59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0,
       44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0,
       41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0,
       42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0,
       55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0,
       59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0,
       44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0,
       49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0,
       42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0,
       55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 55.0, 33.0, 53.0,
       44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0,
       49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0,
       42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0,
       55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 55.0,
       33.0, 53.0, 44.0, 49.0, 71.0, 55.0, 33.0, 53.0, 44.0, 49.0, 71.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0,
       38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0, 52.0, 41.0, 42.0, 53.0, 59.0, 38.0,
       52.0, 41.0, 42.0};
    std::vector<double> kernel_grad_exp = {931., 919., 965., 882., 914., 990., 1051., 1061., 939., 941.,
                                           1015., 897., 931., 919., 965., 882., 914., 990., 1051., 1061.,
                                           939., 941., 1015., 897., 931., 919., 965., 882., 914., 990.,
                                           1051., 1061., 939., 941., 1015., 897., 931., 919., 965., 882.,
                                           914., 990., 1051., 1061., 939., 941., 1015., 897., 931., 919.,
                                           965., 882., 914., 990., 1051., 1061., 939., 941., 1015., 897.};
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAreArray(conv2d_grad_exp));
    ASSERT_THAT(v2.get_grad().to_vector(), ElementsAreArray(kernel_grad_exp));
  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::delete_global_graph();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}