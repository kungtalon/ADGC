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

TEST(OptimizerTest, AdamTest) {
  Graph *graph = Graph::get_instanceof_global_graph();

  try {
    Variable v1 = Variable({2, 2});

    DTensor value1 = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
    DTensor value2 = tensor::Tensor<double>({2, 2}, {-1, 2, -3, 4});
    DTensor value3 = tensor::Tensor<double>({2, 2}, {3, 3, 0, 3});
    DTensor values[3] = {value1, value2, value3};
    DTensor weight = tensor::Tensor<double>({2, 2}, {-1, 2, -1, 2});
    DTensor bias = tensor::Tensor<double>({1}, 2);

    layer::Dense dense_layer(2, 2);
    dense_layer.assign_weight(weight);
    dense_layer.assign_bias(bias);

    // build graph
    auto target = functional::reduce_sum(dense_layer(v1));

    auto optim = optimizer::Adam(target, 0.1);
    optim.set_requires_grads_for_all();

    double weight_expect[12] = {-1.0,
                                1.899999976158142,
                                -1.0,
                                1.899999976158142,
                                -0.9255863428115845,
                                1.9052631855010986,
                                -1.0744136571884155,
                                1.7999999523162842,
                                -0.8680643439292908,
                                1.8789095878601074,
                                -1.1319355964660645,
                                1.6999999284744263};
    double bias_expect[3] = {
      1.899999976158142,
      1.8034818172454834,
      1.7092878818511963,
    };

    std::vector<double> cur_weight;
    for (int ix = 0; ix < 3; ix++) {
      v1.assign_value(values[ix]);
      graph->zero_grad();
      target.forward();
      optim.step();

      ASSERT_THAT(target.get_grad().to_vector(), ElementsAre(1.));
      cur_weight = dense_layer.get_weight().get_value().to_vector();
      for (int j = 0; j < 4; j++) {
        ASSERT_FLOAT_EQ(cur_weight[j], weight_expect[ix * 4 + j])
                  << "cur index : i " + std::to_string(ix) + " j " +
                    std::to_string(j);
      }
      ASSERT_FLOAT_EQ(dense_layer.get_bias().get_value().get_value(),
                      bias_expect[ix])
                << "cur index : i " + std::to_string(ix);
    }

  } catch (const std::exception &ex) {
    FAIL() << "Failed and got this: " << std::endl << ex.what();
  }
  Graph::clear_graph();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}