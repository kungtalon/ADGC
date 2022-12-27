//
// Created by kungtalon on 2022/12/26.
//

#include "autodiff/layer/normalization.h"

namespace auto_diff {
namespace layer {

BatchNorm2D::BatchNorm2D(const size_t &num_channel,
                         const double &epsilon,
                         const double &momentum,
                         Graph *graph)
  : Layer(LayerType::ADG_LAYER_BATCHNORM2D, graph),
    num_channel_(num_channel),
    epsilon_(epsilon),
    momentum_(momentum) {
  if (!num_channel_) {
    throw adg_exception::LayerParameterError("layer >> BatchNorm2D: get invalid num_channel");
  }

  Parameter *gamma_p = new Parameter({num_channel_},
                                     layer_name_ + "_gamma", graph_);
  add_param(gamma_p);

  Parameter *beta_p = new Parameter({num_channel_},
                                    layer_name_ + "_beta", graph_);
  add_param(beta_p);
}

Parameter &BatchNorm2D::get_weight() {
  if (params_ptr_list_.empty()) {
    throw adg_exception::LayerParameterError("No parameters found in layer " +
      layer_name_);
  }
  return *params_ptr_list_[0];
}

Parameter &BatchNorm2D::get_bias() {
  if (params_ptr_list_.empty()) {
    throw adg_exception::LayerParameterError("No parameters found in layer " +
      layer_name_);
  }
  return *params_ptr_list_[1];
}

Node &BatchNorm2D::operator()(const Node &input) {
  check_input(input);

  Node *input_ptr = Graph::get_ptr_of(input.get_full_name(), graph_);

  Parameter &gamma = get_weight();
  Parameter &beta = get_bias();
  bn_node_ptr_ =
    new functional::BatchNorm2D(input_ptr, &gamma, &beta,
                                epsilon_, momentum_, graph_,
                                layer_name_ + "_batchnorm2d");

  return *bn_node_ptr_;
}

void BatchNorm2D::check_input(const Node &input) {
  if (input.get_graph() != graph_) {
    throw adg_exception::MismatchRegisterdGraphError(
      "BatchNorm2D layer " + layer_name_ +
        " does not belong to the same graph as input!");
  }

  // input shape: [B, C, H, W]
  tensor::TensorShape input_shape = input.get_value_shape();
  if (num_channel_ != input_shape[1]) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Invalid input shape for Conv2D layer: expected channel " +
        std::to_string(num_channel_) + " ,got shape " +
        utils::vector_to_str(input_shape));
  }

  if (4 != input_shape.size()) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Invalid input shape for BatchNorm2D layer: expected dimension of 4, got shape " +
        utils::vector_to_str(input_shape));
  }
}

void BatchNorm2D::assign_weight(const DTensor &value) {
  Parameter &gamma = get_weight();
  gamma.assign_value(value);
}

void BatchNorm2D::assign_bias(const DTensor &value) {
  Parameter &beta = get_bias();
  beta.assign_value(value);
}

DTensor BatchNorm2D::get_moving_mean() {
  return bn_node_ptr_->get_moving_mean();
}

DTensor BatchNorm2D::get_moving_var() {
  return bn_node_ptr_->get_moving_var();
}

}
}