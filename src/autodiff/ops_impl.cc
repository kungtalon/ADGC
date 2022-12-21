#include "autodiff/ops.h"

namespace graph_component {
namespace ops {

Add::Add(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
         const std::string &name)
    : Node(NodeType::ADG_ADD_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  // accept two matrices with same size
  // or a matrix along with a value
  if (parents_[0]->get_value_shape() != parents_[1]->get_value_shape()) {
    if (parents_[0]->get_value_size() != 1 &&
        parents_[1]->get_value_size() != 1) {
      throw adg_exception::MismatchNodeValueShapeError("Add >> Add");
    }

    if (parents_[0]->get_value_size() == 1) {
      // let the second shape be of shape 1
      std::iter_swap(parents_.begin(), parents_.begin() + 1);
    }
  }

  value_ = DTensor(parents_[0]->get_value_shape());
};

void Add::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("Add >> do_forward");
  }

  if (parents_[1]->get_value_size() == 1) {
    value_ =
        parents_[0]->get_value().add(parents_[1]->get_value().get_value({0}));
    return;
  }
  value_ = parents_[0]->get_value().add(parents_[1]->get_value());
}

DTensor Add::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("Add >> do_backward");
  }

  if (parents_[1]->get_value_size() == 1 && parent_ptr == parents_[1]) {
    return tensor::Ones({1, get_value_size()});
  }
  return tensor::Eye(get_value_size());
}

VecDot::VecDot(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
               const std::string &name)
    : Node(NodeType::ADG_VECDOT_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  // accept two [n × 1] matrix parents
  if (parents_[0]->get_value().get_dim() != 2 ||
      parents_[0]->get_value_shape()[1] != 1 ||
      parents_[1]->get_value().get_dim() != 2 ||
      parents_[1]->get_value_shape()[1] != 1) {
    throw adg_exception::MismatchNodeValueShapeError("VecDot >> VecDot");
  }

  value_ = DTensor({1});
};

void VecDot::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot >> do_forward");
  }

  DTensor l_mat = parents_[0]->get_value().t();
  value_ = DTensor::dot(l_mat, parents_[1]->get_value());
  value_.reshape({1});
}

DTensor VecDot::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot >> do_backward");
  }

  if (parent_ptr == parents_[0]) {
    return parents_[1]->get_value();
  } else {
    return parents_[0]->get_value();
  }
}

MatMul::MatMul(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
               const std::string &name)
    : Node(NodeType::ADG_MATMUL_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  if (parents_.size() != 2) {
    throw adg_exception::OpsParentsNumException(
        "Matmul >> MatMul: OpsParentsNumException");
  }

  tensor::TensorShape pa_shape_a = parents_[0]->get_value_shape();
  tensor::TensorShape pa_shape_b = parents_[1]->get_value_shape();
  size_t pa_dim_a = pa_shape_a.size();
  size_t pa_dim_b = pa_shape_b.size();
  if (pa_dim_a < 2 || pa_dim_b < 2) {
    throw adg_exception::IncompatibleNodeValueShapeError(
        "Matmul >> MatMul: IncompatibleNodeValueShapeError");
  } else if (pa_shape_a[pa_dim_a - 1] != pa_shape_b[pa_dim_b - 2]) {
    throw adg_exception::IncompatibleNodeValueShapeError(
        "Matmul >> MatMul: IncompatibleNodeValueShapeError");
  }

  value_ =
      DTensor(parent1_ptr->get_value().get_dot_shape(parent2_ptr->get_value()));
};

void MatMul::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("MatMul >> do_forward");
  }

  value_ = DTensor::dot(parents_[0]->get_value(), parents_[1]->get_value());
}

// DTensor MatMul::do_backward(Node *parent_ptr) {
//   if (parents_.empty()) {
//     throw adg_exception::OpsParentsUnsetException("MatMul >> do_backward");
//   }

//   DTensor zeros =
//       tensor::Zeros({get_value_size(), parent_ptr->get_value_size()});
//   if (parent_ptr == parents_[0]) {
//     zeros.fill_diag(parents_[1]->get_value().t().to_vector());
//     zeros.transpose();
//     return zeros;
//   } else {
//     zeros.fill_diag(parents_[0]->get_value().to_vector());
//     tensor::TensorShape row_shape(get_value_shape());
//     tensor::TensorShape col_shape(parent_ptr->get_value_shape());

//     std::reverse(row_shape.begin(), row_shape.end());
//     std::reverse(col_shape.begin(), col_shape.end());

//     std::vector<int32_t> rows_inds_int32 =
//         tensor::Ranges<int32_t>(row_shape, 0).t().to_vector();
//     tensor::TensorSlice rows_inds(rows_inds_int32.begin(),
//                                   rows_inds_int32.end());

//     std::vector<int32_t> cols_inds_int32 =
//         tensor::Ranges<int32_t>(row_shape, 0).t().to_vector();
//     tensor::TensorSlice cols_inds(cols_inds_int32.begin(),
//                                   cols_inds_int32.end());

//     return zeros.take(0, rows_inds).take(1, cols_inds).transpose();
//   }
// }

DTensor MatMul::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("MatMul >> do_backward");
  }

  DTensor zeros =
      tensor::Zeros({get_value_size(), parent_ptr->get_value_size()});
  if (parent_ptr == parents_[0]) {
    // C = A * B
    // jacobi(C, A) = kron(I, B)
    tensor::TensorShape left_shape = parents_[0]->get_value_shape();
    size_t left_row = left_shape[left_shape.size() - 2];
    return DTensor::kron(tensor::Eye(left_row), parents_[1]->get_value());
  } else {
    // jacobi(C, B) = kron(A.T, I)
    tensor::TensorShape right_shape = parents_[1]->get_value_shape();
    size_t right_col = right_shape[right_shape.size() - 1];
    return DTensor::kron(parents_[0]->get_value().t(), tensor::Eye(right_col));
  }
}

MatSum::MatSum(const std::vector<Node *> &parents, Graph *g,
               const std::string &name)
    : Node(NodeType::ADG_MATSUM_TYPE, parents, name, g) {
  if (parents.size() == 0) {
    throw adg_exception::OpsParentsUnsetException(
        "MatSum >> MatSum: get empty parents");
  }

  std::vector<size_t> shape = parents[0]->get_value_shape();
  for (auto parent_ptr : parents) {
    std::vector<size_t> tmp_shape = parent_ptr->get_value_shape();
    if (tmp_shape != shape) {
      throw adg_exception::MismatchNodeValueShapeError(
          "MatSum >> MatSum: inconsistent shape among parents, expected " +
          utils::vector_to_str(shape) + ", but got " +
          utils::vector_to_str(tmp_shape));
    }
  }

  value_ = DTensor(shape);
}

void MatSum::do_forward() {
  value_ = tensor::Zeros(parents_[0]->get_value_shape());
  for (auto parent_ptr : parents_) {
    value_ = value_.add(parent_ptr->get_value());
  }
}

DTensor MatSum::do_backward(Node *parent_ptr) {
  return tensor::Eye(get_value_size());
}

} // namespace ops
} // namespace graph_component
