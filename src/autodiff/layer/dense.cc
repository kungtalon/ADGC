#include "autodiff/layer/layer.h"

namespace auto_diff {

namespace layer {

Dense::Dense(const size_t &input_channel, const size_t &output_channel,
             const std::string &activation, bool use_bias, Graph *graph)
  : Layer(LayerType::ADG_LAYER_DENSE, graph), input_channel_(input_channel),
    activation_(activation) {
  Parameter *kernel_p = new Parameter({input_channel, output_channel},
                                      layer_name_ + "_kernel", graph_);
  add_param(kernel_p);

  if (use_bias) {
    Parameter *bias_p = new Parameter({1}, layer_name_ + "_bias", graph_);
    add_param(bias_p);
  }
}

Node &Dense::operator()(const Node &input) {
  if (input.get_graph() != graph_) {
    throw adg_exception::MismatchRegisterdGraphError(
      "Dense layer " + layer_name_ +
        " does not belong to the same graph as input!");
  }

  tensor::TensorShape input_shape = input.get_value_shape();
  if (input_channel_ != input_shape[input_shape.size() - 1]) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Invalid input shape for Dense layer: expected channel " +
        std::to_string(input_channel_) + " ,got shape " +
        utils::vector_to_str(input_shape));
  }

  Node *input_ptr = Graph::get_ptr_of(input.get_full_name(), graph_);
  Parameter weight = get_weight();
  Node *output =
    new functional::MatMul(input_ptr, &weight, graph_, layer_name_ + "_matmul");

  if (params_ptr_list_.size() == 2) {
    Parameter bias = get_bias();
    output = new functional::Add(output, &bias);
  }

  if (activation_ == "relu") {
    output = new functional::ReLU(output, graph_, layer_name_ + "_relu");
  } else if (activation_ == "sigmoid") {
    output = new functional::Sigmoid(output, graph_, layer_name_ + "_sigmoid");
  } else if (activation_ == "none") {
    // do nothing
  } else {
    throw std::invalid_argument("Dense layer " + layer_name_ +
      " receives invalid activation");
  }

  return *output;
}

Parameter &Dense::get_weight() {
  if (params_ptr_list_.empty()) {
    throw adg_exception::LayerParameterError("No parameters found in layer " +
      layer_name_);
  }
  return *params_ptr_list_[0];
}

Parameter &Dense::get_bias() {
  if (params_ptr_list_.size() <= 1) {
    throw adg_exception::LayerParameterError("No bias in layer " + layer_name_);
  }
  return *params_ptr_list_[1];
}

void Dense::assign_weight(const DTensor &value) {
  Parameter weight = get_weight();
  weight.assign_value(value);
}

void Dense::assign_bias(const DTensor &value) {
  Parameter bias = get_bias();
  bias.assign_value(value);
}

} // namespace layer

} // namespace auto_diff
