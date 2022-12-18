#include "autodiff/functional.h"

namespace graph_component {

namespace functional {

Sigmoid::Sigmoid(const std::vector<Node *> &parents) : Node("add", parents) {
  if (parents.size() != 1) {
    throw adg_exception::FunctionalParentsNumException("Logistic ==> Logistic");
  }
}

void Sigmoid::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
        "Logistic ==> do_forward");
  }
  value_ = parents_[0]->get_value();
  value_.map([](double &val) { val = utils::math::sigmoid(val); });
};

DTensor Sigmoid::do_backward(Node *parent_ptr) {
  DTensor ones = tensor::Ones(get_value_shape());
  DTensor sigmoid_jacob = DTensor::multiply(value_, DTensor::sub(ones, value_));
  return tensor::Diagonal<double>(sigmoid_jacob.to_vector());
}

} // namespace functional

} // namespace graph_component