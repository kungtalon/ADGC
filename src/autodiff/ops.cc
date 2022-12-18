#include "autodiff/ops.h"

namespace graph_component {
namespace ops {

Add::Add(const std::vector<Node *> &parents) : Node("add", parents) {
  // accept two matrices with same size
  if (parents_.size() != 2) {
    throw adg_exception::OpsParentsNumException("Add ==> Add");
  }

  if (parents_[0]->get_value_shape() != parents_[1]->get_value_shape()) {
    throw adg_exception::MismatchNodeValueShapeError("Add ==> Add");
  }
};

void Add::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("Add ==> do_forward");
  }

  value_ = parents_[0]->get_value().add(parents_[1]->get_value());
}

DTensor Add::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("Add ==> do_backward");
  }

  return tensor::Eye(get_value_size());
}

VecDot::VecDot(const std::vector<Node *> &parents) : Node("vecdot", parents) {
  // accept two [n × 1] matrix parents
  if (parents_.size() != 2) {
    throw adg_exception::OpsParentsNumException("VecDot ==> VecDot");
  }

  if (parents_[0]->get_value().get_dim() != 2 ||
      parents_[0]->get_value_shape()[1] != 1 ||
      parents_[1]->get_value().get_dim() != 2 ||
      parents_[1]->get_value_shape()[1] != 2) {
    throw adg_exception::MismatchNodeValueShapeError("VecDot ==> VecDot");
  }
};

void VecDot::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot ==> do_forward");
  }

  DTensor l_mat = parents_[0]->get_value().transpose();
  value_ = l_mat.dot(parents_[1]->get_value());
}

DTensor VecDot::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot ==> do_backward");
  }

  if (parent_ptr == parents_[0]) {
    return parents_[1]->get_value().transpose();
  } else {
    return parents_[0]->get_value().transpose();
  }
}

MatMul::MatMul(const std::vector<Node *> &parents) : Node("matmul", parents) {
  if (parents_.size() != 2) {
    throw adg_exception::OpsParentsNumException("Matmul ==> MatMul");
  }

  tensor::TensorShape pa_shape_a = parents_[0]->get_value_shape();
  tensor::TensorShape pa_shape_b = parents_[1]->get_value_shape();
  size_t pa_dim_a = pa_shape_a.size();
  size_t pa_dim_b = pa_shape_b.size();
  if (pa_dim_a < 2 || pa_dim_b < 2) {
    throw adg_exception::IncompatibleNodeValueShapeError("Matmul ==> MatMul");
  } else if (pa_shape_a[pa_dim_a - 1] != pa_shape_b[pa_dim_b - 2] ||
             pa_shape_a[pa_dim_a - 2] != pa_shape_b[pa_dim_b - 1]) {
    throw adg_exception::IncompatibleNodeValueShapeError("Matmul ==> MatMul");
  }
};

void MatMul::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("MatMul ==> do_forward");
  }

  value_ = parents_[0]->get_value().dot(parents_[1]->get_value());
}

DTensor MatMul::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("MatMul ==> do_backward");
  }

  DTensor zeros = tensor::Zeros({get_value_size(), get_value_size()});
  if (parent_ptr == parents_[0]) {
  }
}

} // namespace ops
} // namespace graph_component