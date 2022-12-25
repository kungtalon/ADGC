//
// Created by kungtalon on 2022/12/25.
//

#include "autodiff/component/functional/activation.h"

namespace auto_diff {

namespace functional {

// class implementations:
//

Sigmoid::Sigmoid(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_SIGMOID_TYPE, {parent_ptr}, name, g) {
  set_backward_version(1);
  value_ = DTensor(parent_ptr->get_value_shape());
}

void Sigmoid::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "Logistic >> do_forward");
  }
  value_ = parents_[0]->get_value().copy();
  value_.map([](double &val) { val = utils::math::sigmoid(val); });
};

DTensor Sigmoid::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "Sigmoid >> do_backward");
  }

  DTensor ones = tensor::Ones(get_value_shape());
  DTensor sigmoid_backward =
    tensor::multiply(value_, tensor::sub(ones, value_));
  return sigmoid_backward.multiply(get_grad());
}

ReLU::ReLU(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_RELU_TYPE, {parent_ptr}, name, g) {
  set_backward_version(1);
  value_ = DTensor(parent_ptr->get_value_shape());
}

void ReLU::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException("ReLU >> do_forward");
  }

  value_ = parents_[0]->get_value().copy();
  value_.map([](double &val) { val = utils::math::relu(val); });
}

DTensor ReLU::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException("ReLU >> do_backward");
  }

  DTensor relu_backward = value_.copy();
  relu_backward.map([](double &val) {
    if (val > 0.0) {
      val = 1.0;
    } else {
      val = 0.0;
    }
  });
  return relu_backward.multiply(get_grad());
}

//
// function implementations:

Sigmoid &sigmoid(const Node &parent, Graph *g, const std::string &name) {
  Sigmoid *node_ptr =
    new Sigmoid(Graph::get_ptr_of(parent.get_full_name(), g), g, name);
  return *node_ptr;
}

ReLU &relu(const Node &parent, Graph *g, const std::string &name) {
  ReLU *node_ptr =
    new ReLU(Graph::get_ptr_of(parent.get_full_name(), g), g, name);
  return *node_ptr;
}

}

}