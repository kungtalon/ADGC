//
// Created by kungtalon on 2022/12/25.
//
#include "autodiff/component/functional/loss.h"

namespace auto_diff {

namespace functional {

// class implementations:

// loss function takes two parents, one is the label data being Variable type
CrossEntropyWithSoftMax::CrossEntropyWithSoftMax(Node *parent_ptr,
                                                 Variable *labels_ptr, Graph *g,
                                                 const std::string &name)
  : Node(NodeType::ADG_CROSS_ENTROPY_SOFTMAX_TYPE, {parent_ptr, labels_ptr},
         name, g) {
  set_backward_version(1);
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
    tensor::sum(tensor::multiply(parents_[1]->get_value(), neg_log_probs_));
}

DTensor CrossEntropyWithSoftMax::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
      "CrossEntropyWithSoftMax >> do_backward");
  }

  DTensor result;
  if (parent_ptr == parents_[0]) {
    result = tensor::sub(probs_, parents_[1]->get_value());
  } else {
    result = neg_log_probs_;
  }
  // throw adg_exception::TestingDebugException(
  //     "Get pa value of " +
  //     utils::vector_to_str(parent_ptr->get_value().to_vector()));
  return result.multiply(get_grad().get_value());
}

DTensor CrossEntropyWithSoftMax::get_probs() {
  if (this != unique_ptr_) {
    auto real_ptr = dynamic_cast<CrossEntropyWithSoftMax *>(unique_ptr_);
    return real_ptr->probs_.copy();
  }

  return probs_.copy();
}


// function implementations:

ReduceSum &reduce_sum(const Node &input, Graph *g, const std::string &name) {
  ReduceSum *node_ptr =
    new ReduceSum(Graph::get_ptr_of(input.get_full_name(), g), g, name);
  return *node_ptr;
}

ReduceMean &reduce_mean(const Node &input, Graph *g, const std::string &name) {
  ReduceMean *node_ptr =
    new ReduceMean(Graph::get_ptr_of(input.get_full_name(), g), g, name);
  return *node_ptr;
}

}
}