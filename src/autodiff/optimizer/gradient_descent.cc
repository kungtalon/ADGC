#include "autodiff/optimizer/gradient_descent.h"

namespace auto_diff {
namespace optimizer {

GradientDescent::GradientDescent(const Node &target,
                                 const double &learning_rate, Graph *graph)
  : Optimizer(target, learning_rate, graph) {}

void GradientDescent::update() {
  for (auto *node_ptr : trainable_params_list_) {
    if (node_ptr->get_type() == NodeType::ADG_PARAMETER_TYPE) {
      DTensor grad = get_gradient(node_ptr);

      DTensor value = node_ptr->get_value(); // shallow copy of value tensor
      value -= grad.multiply(learning_rate_);
    }
  }
}

} // namespace optimizer
} // namespace auto_diff