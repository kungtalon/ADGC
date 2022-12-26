//
// Created by kungtalon on 2022/12/25.
//
#include "autodiff/component/functional/manipulation.h"

namespace auto_diff {
namespace functional {

Reshape::Reshape(Node *parent_ptr, const tensor::TensorShape &shape, Graph *g,
                 const std::string &name)
  : Node(NodeType::ADG_RESHAPE_TYPE, {parent_ptr}, name, g) {
  set_backward_version(1);
  size_t new_size = 1;
  for (auto len : shape) {
    new_size *= len;
  }

  if (parents_[0]->get_value_size() != new_size) {
    throw adg_exception::MismatchNodeValueShapeError("Reshape >> Reshape: MismatchNodeValueShapeError for reshape");
  }

  value_ = DTensor(shape);
  new_shape_ = shape;
}

void Reshape::do_forward() {
  value_ = parents_[0]->get_value().copy();
  value_.reshape(new_shape_);
}

DTensor Reshape::do_backward(Node *parent_ptr) {
  return get_grad(false);
}

Pad2D::Pad2D(Node *parent_ptr, const std::vector<std::pair<size_t, size_t>> &padding, const double &value,
             Graph *g, const std::string &name)
  : Node(NodeType::ADG_PAD2D_TYPE, {parent_ptr}, name, g), pad_value_(value), padding_(padding) {
  set_backward_version(1);
  if (padding.size() != 2) {
    throw adg_exception::InvalidNodeArgumentError(
      "Pad2D >> Pad2D: expect padding for 2 dimensions, got " + std::to_string(padding.size()));
  }

  tensor::TensorShape shape = parent_ptr->get_value_shape();
  size_t dim = shape.size();
  shape[dim - 2] += padding[0].first + padding[0].second;
  shape[dim - 1] += padding[1].first + padding[1].second;
  value_ = DTensor(shape);
}

void Pad2D::do_forward() {
  value_ = tensor::pad2d(parents_[0]->get_value(), padding_, pad_value_);
}

DTensor Pad2D::do_backward(Node *parent_ptr) {
  size_t pad_left, pad_right, pad_top, pad_bottom;
  size_t dim = get_value_dim();
  tensor::TensorShape shape = get_value_shape();

  pad_top = padding_[0].first;
  pad_bottom = padding_[0].second;
  pad_left = padding_[1].first;
  pad_right = padding_[1].second;

  DTensor grad = get_grad();
  return grad.slice({{dim - 2, pad_top, shape[dim - 2] - pad_bottom}, {dim - 1, pad_left, shape[dim - 1] - pad_right}});
}

}
}