//
// Created by kungtalon on 2022/12/26.
//

#include "autodiff/layer/convolution.h"

namespace auto_diff {
namespace layer {

Conv2D::Conv2D(const size_t &input_channel,
               const size_t &output_channel,
               const std::array<size_t, 2> &kernel_size,
               const std::array<size_t, 2> &stride,
               const std::array<size_t, 2> &padding,
               const std::string &activation,
               bool use_bias,
               Graph *graph) : Layer(LayerType::ADG_LAYER_CONV2D, graph),
                               input_channel_(input_channel),
                               output_channel_(output_channel),
                               padding_({padding[0], padding[0], padding[1], padding[1]}),
                               stride_(stride) {
  if (!kernel_size[0] || !kernel_size[1]) {
    throw adg_exception::LayerParameterError("Conv2D receives a 0 kernel size...");
  }

  if (!stride_[0] || !stride_[1]) {
    throw adg_exception::LayerParameterError("Conv2D receives a 0 stride...");
  }

  Parameter *kernel_p = new Parameter({output_channel_, input_channel_, kernel_size[0], kernel_size[1]},
                                      layer_name_ + "_kernel", graph_);
  add_param(kernel_p);

  if (use_bias) {
    Parameter *bias_p = new Parameter({output_channel_},
                                      layer_name_ + "_bias", graph_);
    add_param(bias_p);
  }

  set_config("activation", activation);
}

Conv2D::Conv2D(const size_t &input_channel,
               const size_t &output_channel,
               const std::array<size_t, 2> &kernel_size,
               const std::array<size_t, 2> &stride,
               const std::string_view &padding,
               const std::string &activation,
               bool use_bias,
               Graph *graph) : Layer(LayerType::ADG_LAYER_CONV2D, graph),
                               input_channel_(input_channel),
                               output_channel_(output_channel),
                               stride_(stride) {
  if (!kernel_size[0] || !kernel_size[1]) {
    throw adg_exception::LayerParameterError("Conv2D receives a 0 kernel size...");
  }

  if (!stride_[0] || !stride_[1]) {
    throw adg_exception::LayerParameterError("Conv2D receives a 0 stride...");
  }

  if (padding != "SAME" && padding != "VALID") {
    throw adg_exception::LayerParameterError("Conv2D padding should be 'SAME' or 'VALID'...");
  }

  if (padding == "VALID") {
    // 'VALID' means no padding
    padding_ = {0, 0, 0, 0};
  } else {
    // 'SAME' means keeping the same size after conv (only when stride=1)
    // namely, out_h = h - k + 2p + 1 = h ==> p = ⌊(k-1)/2⌋
    padding_ = {kernel_size[0] / 2, kernel_size[0] / 2, kernel_size[1] / 2, kernel_size[1] / 2};
    // if k is even, pad extra one to the right/bottom
    if (kernel_size[0] & 1 == 0) {
      padding_[1]++;
    }
    if (kernel_size[1] & 1 == 0) {
      padding_[3]++;
    }
  }

  Parameter *kernel_p = new Parameter({output_channel_, input_channel_, kernel_size[0], kernel_size[1]},
                                      layer_name_ + "_kernel", graph_);
  add_param(kernel_p);

  if (use_bias) {
    Parameter *bias_p = new Parameter({output_channel_},
                                      layer_name_ + "_bias", graph_);
    add_param(bias_p);
  }

  set_config("activation", activation);
}

Parameter &Conv2D::get_weight() {
  if (params_ptr_list_.empty()) {
    throw adg_exception::LayerParameterError("No parameters found in layer " +
      layer_name_);
  }
  return *params_ptr_list_[0];
}

Parameter &Conv2D::get_bias() {
  if (params_ptr_list_.size() <= 1) {
    throw adg_exception::LayerParameterError("No bias in layer " + layer_name_);
  }
  return *params_ptr_list_[1];
}

Node &Conv2D::operator()(const Node &input) {
  check_input(input);

  Node *input_ptr = Graph::get_ptr_of(input.get_full_name(), graph_);
  Node *output_ptr = input_ptr;

  if (padding_ != std::array<size_t, 4>({0, 0, 0, 0})) {
    output_ptr = new functional::Pad2D(output_ptr,
                                       {{padding_[0], padding_[1]}, {padding_[2], padding_[3]}});
  }

  Parameter weight = get_weight();
  output_ptr =
    new functional::Conv2D(output_ptr, &weight, stride_, graph_, layer_name_ + "_conv2d");

  if (params_ptr_list_.size() == 2) {
    Parameter bias = get_bias();
    output_ptr = new functional::MatAddVec(output_ptr, &bias, 1);
  }

  output_ptr = use_activation(output_ptr);

  return *output_ptr;
}

void Conv2D::check_input(const Node &input) {
  if (input.get_graph() != graph_) {
    throw adg_exception::MismatchRegisterdGraphError(
      "Conv2D layer " + layer_name_ +
        " does not belong to the same graph as input!");
  }

  // input shape: [B, C, H, W]
  tensor::TensorShape input_shape = input.get_value_shape();
  if (input_channel_ != input_shape[1]) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Invalid input shape for Conv2D layer: expected channel " +
        std::to_string(input_channel_) + " ,got shape " +
        utils::vector_to_str(input_shape));
  }

  if (4 != input_shape.size()) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Invalid input shape for Conv2D layer: expected dimension of 4, got shape " +
        utils::vector_to_str(input_shape));
  }
}

void Conv2D::assign_weight(const DTensor &value) {
  Parameter weight = get_weight();
  weight.assign_value(value);
}

void Conv2D::assign_bias(const DTensor &value) {
  Parameter bias = get_bias();
  bias.assign_value(value);
}

}
}