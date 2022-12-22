#include "autodiff/optimizer/gradient_descent.h"

namespace auto_diff {
namespace optimizer {

GradientDescent::GradientDescent(const Node &target, const size_t &batch_size,
                                 const double &learning_rate, Graph *graph)
    : Optimizer(target, batch_size, graph), learning_rate_(learning_rate) {}

void GradientDescent::update() {
  NodeIteratorPair node_iterators = graph_->get_node_iterators();

  for (auto node_iter = node_iterators.first;
       node_iter != node_iterators.second; ++node_iter) {
    Node *node_ptr = *node_iter;
    if (node_ptr->get_type() != NodeType::ADG_PARAMETER_TYPE) {
      continue;
    }

    Parameter *param_ptr = static_cast<Parameter *>(node_ptr);
    if (!param_ptr->is_trainable()) {
      continue;
    }

    DTensor grad = get_gradient(node_ptr);

    DTensor value = node_ptr->get_value(); // shallow copy of value tensor
    value -= grad.multiply(learning_rate_);
  }
}

} // namespace optimizer
} // namespace auto_diff