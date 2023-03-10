#include "autodiff/optimizer/adam.h"

namespace auto_diff {
namespace optimizer {
Adam::Adam(const Node &target,
           const double &learning_rate, const double &beta1,
           const double &beta2, const double &weight_decay,
           const double &epsilon, Graph *graph)
  : Optimizer(target, learning_rate, graph),
    weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2),
    beta1_power_(1.), beta2_power_(1.), epsilon_(epsilon) {}

void Adam::reset_state() {
  beta1_power_ = 1.;
  beta2_power_ = 1.;
  moments_table_.clear();
}

void Adam::update() {
  for (auto *node_ptr : trainable_params_list_) {
    if (node_ptr->get_type() == NodeType::ADG_PARAMETER_TYPE) {
      DTensor grad = get_gradient(node_ptr);

      if (weight_decay_ >= 0) {
        // g = g + lambda * theta
        grad += node_ptr->get_value().multiply(weight_decay_);
      }

      DTensor moment = update_moments(node_ptr->get_full_name(), grad);

      DTensor value = node_ptr->get_value(); // shallow copy of value tensor
      value -= moment.multiply(learning_rate_);
    }
  }

  beta1_power_ *= beta1_;
  beta2_power_ *= beta2_;
}

DTensor Adam::update_moments(const std::string &node_name,
                             const DTensor &grad) {
  std::string first_moment_key = node_name + ":FIRST";
  std::string second_moment_key = node_name + ":SECOND";

  DTensor first_moment, second_moment;
  if (moments_table_.find(first_moment_key) == moments_table_.end()) {
    first_moment = DTensor(grad.get_shape());
  } else {
    first_moment = moments_table_[first_moment_key].multiply(beta1_);
  }
  if (moments_table_.find(second_moment_key) == moments_table_.end()) {
    second_moment = DTensor(grad.get_shape());
  } else {
    second_moment = moments_table_[second_moment_key].multiply(beta2_);
  }

  // m = beta * m + (1 - beta) * g
  first_moment += grad.multiply(1 - beta1_);
  second_moment += grad.multiply(grad).multiply(1 - beta2_);
  moments_table_[first_moment_key] = first_moment;
  moments_table_[second_moment_key] = second_moment;

  // normalize the moving average moments
  first_moment = first_moment.multiply(1. / (1 - beta1_power_ * beta1_));
  second_moment = second_moment.multiply(1. / (1 - beta2_power_ * beta2_));
  second_moment += epsilon_;
  second_moment = tensor::sqrt(second_moment);

  // get the self-adaptive gradients
  return tensor::div(first_moment, second_moment);
}

} // namespace optimizer
} // namespace auto_diff