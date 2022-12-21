#include "autodiff/layer/layers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace testing;
using namespace graph_component;

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}