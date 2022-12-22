#define TENSOR_TESTING true
#define ENABLE_TENSOR_MULTI_THREAD true

#include "autodiff/component/node.h"
#include "autodiff/graph.h"
#include "autodiff/layer/layer.h"
#include "autodiff/optimizer/optimizer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace auto_diff;

TEST(LayerTest, GradientDescentTest) {
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({2, 2});

    DTensor value1 = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
    DTensor value2 = tensor::Tensor<double>({2, 2}, {-1, 2, -3, 4});
    DTensor weight = tensor::Tensor<double>({2, 2}, {-1, 2, -1, 2});
    DTensor bias = tensor::Tensor<double>({1}, 2);

    layer::Dense dense_layer(2, 2);
    dense_layer.assign_weight(weight);
    dense_layer.assign_bias(bias);

    // build graph
    auto target = functional::reduce_sum(dense_layer(v1));

    auto optim = optimizer::GradientDescent(target, 2, 0.1);

    // step one
    v1.assign_value(value1);
    graph->zero_grad();
    target.forward();
    optim.step();

    // test forward
    ASSERT_EQ(v1.get_value_shape(), tensor::TensorShape({2, 2}));
    ASSERT_EQ(dense_layer.get_weight().get_value_shape(),
              tensor::TensorShape({2, 2}));
    ASSERT_THAT(dense_layer.get_weight().get_value().to_vector(),
                ElementsAre(-1., 2., -1, 2));
    ASSERT_FLOAT_EQ(target.get_value().get_value(), 24.);

    // test backward
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(2., 2., 2., 2.));
    ASSERT_THAT(dense_layer.get_weight().get_grad().to_vector(),
                ElementsAre(0., 4., 0., 6.));
    ASSERT_FLOAT_EQ(dense_layer.get_bias().get_grad().get_value(), 2.);

    // step 2, should update value
    v1.assign_value(value2);
    graph->zero_grad();
    target.forward();
    optim.step();

    // check grad
    ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
    ASSERT_EQ(v1.get_grad().get_shape(), tensor::TensorShape({2, 2}));
    ASSERT_THAT(v1.get_grad().to_vector(), ElementsAre(1., 1., 1., 1.));
    ASSERT_THAT(dense_layer.get_weight().get_grad().to_vector(),
                ElementsAre(-4., -4, 6., 6.));
    ASSERT_FLOAT_EQ(dense_layer.get_bias().get_grad().get_value(), 4.);

    // check updated value
    ASSERT_THAT(v1.get_value().to_vector(),
                ElementsAre(-1, 2, -3, 4)); // v1 should not change
    ASSERT_THAT(dense_layer.get_weight().get_value().to_vector(),
                ElementsAre(-0.8, 2.0, -1.3, 1.4));
    ASSERT_THAT(dense_layer.get_bias().get_value().to_vector(),
                ElementsAre(1.70));

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