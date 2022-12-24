#include "autodiff/component/functional.h"

namespace auto_diff {

namespace functional {

Sigmoid::Sigmoid(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_SIGMOID_TYPE, {parent_ptr}, name, g) {
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
    DTensor::multiply(value_, DTensor::sub(ones, value_));
  return tensor::Diagonal<double>(
    sigmoid_backward.to_vector()); // shape [value_size, value_size]
}

ReLU::ReLU(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_RELU_TYPE, {parent_ptr}, name, g) {
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
  return tensor::Diagonal<double>(
    relu_backward.to_vector()); // shape: [value_size, value_size]
}

// loss function takes two parents, one is the label data being Variable type
CrossEntropyWithSoftMax::CrossEntropyWithSoftMax(Node *parent_ptr,
                                                 Variable *labels_ptr, Graph *g,
                                                 const std::string &name)
  : Node(NodeType::ADG_CROSS_ENTROPY_SOFTMAX_TYPE, {parent_ptr, labels_ptr},
         name, g) {
  value_ = DTensor({1});
}

DTensor CrossEntropyWithSoftMax::softmax(const DTensor &input) {
  DTensor output = input.copy();

  output.map(
    [](double &val) { val = std::exp(std::min(val, 100.0)); }); // [N, d]
  size_t dim = output.get_dim();
  size_t ncol = output.get_shape()[dim - 1];

  DTensor exp_sum = output.sum(dim - 1).add(epsilon_); // [N]
  output.map(tensor::Mapper<double>(
    [&exp_sum, &ncol](double &val, const size_t &index) {
      size_t row_id = index / ncol;
      val /= exp_sum.get_value({row_id});
    }));
  return output;
}

void CrossEntropyWithSoftMax::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "CrossEntropyWithSoftMax >> do_forward");
  }

  probs_ = softmax(parents_[0]->get_value()); // shape: [N, D]
  neg_log_probs_ = probs_.copy();
  neg_log_probs_.map([](double &val) { val = -std::log(val + epsilon_); });
  // sum_i { - yi * log(pi) }
  value_ =
    DTensor::sum(DTensor::multiply(parents_[1]->get_value(), neg_log_probs_));
}

DTensor CrossEntropyWithSoftMax::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "CrossEntropyWithSoftMax >> do_backward");
  }

  DTensor result;
  if (parent_ptr == parents_[0]) {
    result = DTensor::sub(probs_, parents_[1]->get_value());
  } else {
    result = neg_log_probs_;
  }
  // throw adg_exception::TestingDebugException(
  //     "Get pa value of " +
  //     utils::vector_to_str(parent_ptr->get_value().to_vector()));
  result.reshape({parent_ptr->get_value_size(), 1});
  return result;
}

DTensor CrossEntropyWithSoftMax::get_probs() {
  if (this != unique_ptr_) {
    auto real_ptr = dynamic_cast<CrossEntropyWithSoftMax *>(unique_ptr_);
    return real_ptr->probs_.copy();
  }
  
  return probs_.copy();
}

ReduceSum::ReduceSum(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_REDUCE_SUM_TYPE, {parent_ptr}, name, g) {
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
  return tensor::Ones({parent_ptr->get_value_size(), 1});
}

ReduceMean::ReduceMean(Node *parent_ptr, Graph *g, const std::string &name)
  : Node(NodeType::ADG_REDUCE_SUM_TYPE, {parent_ptr}, name, g) {
  value_ = DTensor({1});
}

void ReduceMean::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "ReduceMean >> ReduceMean: FunctionalParentsUnsetException");
  }

  multiplier_ = 1 / parents_[0]->get_value_size();
  value_ = parents_[0]->get_value().sum().multiply(multiplier_);
}

DTensor ReduceMean::do_backward(Node *parent_ptr) {
  return DTensor({parent_ptr->get_value_size(), 1}, multiplier_);
}

} // namespace functional

} // namespace auto_diff