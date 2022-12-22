#include "autodiff/optimizer/optimizer.h"

namespace auto_diff {
namespace optimizer {

Optimizer::Optimizer(const Node &target, const size_t &batch_size,
                     const double &learning_rate, Graph *graph)
    : batch_size_(batch_size), acc_counter_(0), learning_rate_(learning_rate) {
  if (graph == nullptr) {
    graph_ = Graph::get_instanceof_global_graph();
  } else {
    graph_ = graph;
  }
  target_node_ptr_ = Graph::get_ptr_of(target.get_full_name(), graph_);
}

void Optimizer::zero_grad() { graph_->zero_grad(); }

void Optimizer::step() {
  propagate();

  acc_counter_++;
  if (acc_counter_ >= batch_size_) {
    update();
    acc_grads_.clear();
    acc_counter_ = 0;
  }
}

bool Optimizer::is_trainable_param(Node *node_ptr) {
  if (node_ptr->get_type() != NodeType::ADG_PARAMETER_TYPE) {
    return false;
  }

  Parameter *param_ptr = static_cast<Parameter *>(node_ptr);
  if (!param_ptr->is_trainable()) {
    return false;
  }

  return true;
}

DTensor Optimizer::get_gradient(Node *node_ptr) {
  return acc_grads_.at(node_ptr->get_full_name()) / batch_size_;
}

void Optimizer::propagate() {
  // backward is done here
  // node_iterators: pair of <begin_iterator, end_iterator>
  NodeIteratorPair node_iterators = graph_->get_node_iterators();

  for (auto node_iter = node_iterators.first;
       node_iter != node_iterators.second; ++node_iter) {
    Node *node_ptr = *node_iter;

    if (node_ptr->get_type() != NodeType::ADG_VARIABLE_TYPE &&
        node_ptr->get_type() != NodeType::ADG_PARAMETER_TYPE) {
      continue;
    }

    node_ptr->backward(target_node_ptr_);
    DTensor grad = node_ptr->get_grad();
    auto acc_grads_iter = acc_grads_.find(node_ptr->get_full_name());
    if (acc_grads_iter == acc_grads_.end()) {
      acc_grads_[node_ptr->get_full_name()] = grad;
    } else {
      acc_grads_iter->second += grad;
    }
  }
}
} // namespace optimizer

} // namespace auto_diff
