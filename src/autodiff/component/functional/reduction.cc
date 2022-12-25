//
// Created by kungtalon on 2022/12/25.
//

#include "autodiff/component/functional/reduction.h"

namespace auto_diff {

namespace functional {

// class implementations:

ReduceSum::ReduceSum(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_REDUCE_SUM_TYPE, {parent_ptr}, name, g) {
  set_backward_version(1);
  value_ = DTensor({1});
}

void ReduceSum::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "ReduceSum >> ReduceSum");
  }

  value_ = parents_[0]->get_value().sum();
}

DTensor ReduceSum::do_backward(Node *parent_ptr) {
  return DTensor({parent_ptr->get_value_size(), 1}, get_grad().get_value());
}

ReduceMean::ReduceMean(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_REDUCE_SUM_TYPE, {parent_ptr}, name, g) {
  set_backward_version(1);
  value_ = DTensor({1});
}

void ReduceMean::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "ReduceMean >> ReduceMean: FunctionalParentsUnsetException");
  }

  multiplier_ = 1. / parents_[0]->get_value_size();
  value_ = parents_[0]->get_value().sum().multiply(multiplier_);
}

DTensor ReduceMean::do_backward(Node *parent_ptr) {
  return DTensor({parent_ptr->get_value_size(), 1}, multiplier_ * get_grad().get_value());
}

// function implementations:
CrossEntropyWithSoftMax &cross_entropy_with_softmax(const Node &input,
                                                    const Variable &labels,
                                                    Graph *g,
                                                    const std::string &name) {
  Node *parent_ptr = Graph::get_ptr_of(input.get_full_name(), g);
  Variable *label_ptr =
    dynamic_cast<Variable *>(Graph::get_ptr_of(labels.get_full_name(), g));
  CrossEntropyWithSoftMax *node_ptr =
    new CrossEntropyWithSoftMax(parent_ptr, label_ptr, g, name);
  return *node_ptr;
}

}
}