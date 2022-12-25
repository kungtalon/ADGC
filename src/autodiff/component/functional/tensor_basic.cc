//
// Created by kungtalon on 2022/12/25.
//
#include "autodiff/component/functional/tensor_basic.h"

namespace auto_diff {
namespace functional {

Add::Add(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
         const std::string &name)
  : Node(NodeType::ADG_ADD_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  set_backward_version(1);
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
  if (parents_[1]->get_value_size() == 1) {
    value_ =
      parents_[0]->get_value().add(parents_[1]->get_value().get_value({0}));
    return;
  }
  value_ = parents_[0]->get_value().add(parents_[1]->get_value());
}

DTensor Add::do_backward(Node *parent_ptr) {
  if (parents_[1]->get_value_size() == 1 && parent_ptr == parents_[1]) {
//    return tensor::Ones({1, get_value_size()});
    return jacobi_.sum();;
  }
//  return tensor::Eye(get_value_size());
  return jacobi_;
}

MatAddVec::MatAddVec(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
                     const std::string &name)
  : Node(NodeType::ADG_MATADDVEC_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  // accept one matrix and one vector with same size in the last dim
  tensor::TensorShape matrix_shape = parent1_ptr->get_value_shape();
  tensor::TensorShape vector_shape = parent2_ptr->get_value_shape();
  if (matrix_shape.size() < 2 || vector_shape.size() != 1) {
    throw adg_exception::MismatchNodeValueShapeError("Add >> Add");
  }

  if (matrix_shape[matrix_shape.size() - 1] != vector_shape[0]) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Add >> Add: Matrix and vector don't have same size in the last dimension...");
  }

  value_ = DTensor(parents_[0]->get_value_shape());
};

void MatAddVec::do_forward() {
  auto matrix_values = parents_[0]->get_value().to_vector();
  auto vector_values = parents_[1]->get_value().to_vector();
  size_t repeat_times = matrix_values.size() / vector_values.size();

  for (int ix = 0; ix < vector_values.size(); ++ix) {
    cblas_daxpy(repeat_times, 1.0, &vector_values[ix], 0, &matrix_values[ix], vector_values.size());
  }
  value_ = DTensor(parents_[0]->get_value_shape(), matrix_values);
}

DTensor MatAddVec::do_backward(Node *parent_ptr) {
  if (parent_ptr == parents_[0]) {
    return tensor::Eye(get_value_size());
  }

  size_t repeat_times = parents_[0]->get_value_size() / parents_[1]->get_value_size();
  return DTensor::kron(tensor::Ones({1, repeat_times}), tensor::Eye(parents_[1]->get_value_size()));
}

VecDot::VecDot(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
               const std::string &name)
  : Node(NodeType::ADG_VECDOT_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  set_backward_version(1);
  // accept two [n × 1] matrix parents
  if (parents_[0]->get_value().get_dim() != 2 ||
    parents_[0]->get_value_shape()[1] != 1 ||
    parents_[1]->get_value().get_dim() != 2 ||
    parents_[1]->get_value_shape()[1] != 1) {
    throw adg_exception::MismatchNodeValueShapeError("VecDot >> VecDot");
  }

  value_ = DTensor({1});
}

void VecDot::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot >> do_forward");
  }

  DTensor l_mat = parents_[0]->get_value().t();
  value_ = tensor::dot(l_mat, parents_[1]->get_value());
  value_.reshape({1});
}

DTensor VecDot::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("VecDot >> do_backward");
  }

  if (parent_ptr == parents_[0]) {
    return parents_[1]->get_value().multiply(get_grad().get_value());
  } else {
    return parents_[0]->get_value().multiply(get_grad().get_value());
  }
}

MatMul::MatMul(Node *parent1_ptr, Node *parent2_ptr, Graph *g,
               const std::string &name)
  : Node(NodeType::ADG_MATMUL_TYPE, {parent1_ptr, parent2_ptr}, name, g) {
  set_backward_version(1);
  // limitation: the second matrix has to be in rank 2.
  // the first tensor can have any rank no less than 2.
  tensor::TensorShape pa_shape_a = parents_[0]->get_value_shape();
  tensor::TensorShape pa_shape_b = parents_[1]->get_value_shape();
  size_t pa_dim_a = pa_shape_a.size();
  size_t pa_dim_b = pa_shape_b.size();
  if (pa_dim_b != 2) {
    throw adg_exception::IncompatibleNodeValueShapeError(
      "Matmul >> MatMul: IncompatibleNodeValueShapeError");
  } else if (pa_shape_a[pa_dim_a - 1] != pa_shape_b[pa_dim_b - 2]) {
    throw adg_exception::IncompatibleNodeValueShapeError(
      "Matmul >> MatMul: IncompatibleNodeValueShapeError");
  }

  value_ =
    DTensor(parent1_ptr->get_value().get_dot_shape(parent2_ptr->get_value()));
}

void MatMul::do_forward() {
  value_ = tensor::dot(parents_[0]->get_value(), parents_[1]->get_value());
}

DTensor MatMul::do_backward(Node *parent_ptr) {
  if (parents_.empty()) {
    throw adg_exception::OpsParentsUnsetException("MatMul >> do_backward");
  }

//  // aggregate the batch dim with the row dim... it is equivalent
//  DTensor left = parents_[0]->get_value();
//  size_t left_last_len = left.get_shape()[left.get_dim() - 1];
//  left.reshape({left.get_size() / left_last_len, left_last_len});
//  if (parent_ptr == parents_[0]) {
//    // C = A * B
//    // jacobi(C, A) = kron(I, B)
//    tensor::TensorShape left_shape = parents_[0]->get_value_shape();
//    size_t left_row = left.get_shape()[0];
//    return DTensor::kron(tensor::Eye(left_row), parents_[1]->get_value());
//  } else {
//    // jacobi(C, B) = kron(A.T, I)
//    tensor::TensorShape right_shape = parents_[1]->get_value_shape();
//    size_t right_col = right_shape[right_shape.size() - 1];
//    return DTensor::kron(left.t(), tensor::Eye(right_col));
//  }
  DTensor res;
  DTensor grad = get_grad();
  if (parent_ptr == parents_[0]) {
    // A [M, N]
    // B [N, K]
    // grad [M, K]
    // dA = grad dot B.T
    res = grad.dot(parents_[1]->get_value().t());
  } else {
    // dB = A.T dot grad
//    DTensor grad = get_grad();
//    DTensor left_tensor = parents_[0]->get_value();
//    size_t left_ncols = parents_[1]->get_value_shape()[0];
//    size_t grad_ncols = parents_[1]->get_value_shape()[1];
//    left_tensor.reshape({left_tensor.get_size() / left_ncols, left_ncols});
//    grad.reshape({grad.get_size() / grad_ncols, grad_ncols});
//    res = left_tensor.t().dot(grad);   // [3, 2, 3]
    // dB = A.T dot grad
    size_t left_dim = parents_[0]->get_value_dim();
    res = parents_[0]->get_value().transpose(left_dim - 1, left_dim - 2).dot(grad); // [3, 2, 3]
    while (res.get_dim() > 2) {
      res = res.sum(0);
    }
  }
  return res;
}

MatSum::MatSum(const std::vector<Node *> &parents, Graph *g,
               const std::string &name)
  : Node(NodeType::ADG_MATSUM_TYPE, parents, name, g) {
  set_backward_version(1);
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
  return get_grad(false);
}

PointMul::PointMul(Node *parent_ptr1, Node *parent_ptr2, Graph *g, const std::string &name)
  : Node(NodeType::ADG_POINTMUL_TYPE, {parent_ptr1, parent_ptr2}, name, g) {
  set_backward_version(1);
  if (parents_[0]->get_value_shape() != parents_[1]->get_value_shape()) {
    throw adg_exception::MismatchNodeValueShapeError("PointMul >> PointMul: MismatchNodeValueShapeError");
  }

  value_ = DTensor(parents_[0]->get_value_shape());
}

void PointMul::do_forward() {
  value_ = parents_[0]->get_value().multiply(parents_[1]->get_value());
}

DTensor PointMul::do_backward(Node *parent_ptr) {
  if (parent_ptr == parents_[0]) {
    return parents_[1]->get_value().multiply(get_grad());
  } else {
    return parents_[0]->get_value().multiply(get_grad());
  }
}


// functions:

Add &add(const Node &parent1, const Node &parent2, Graph *g,
         const std::string &name) {
  Add *node_ptr =
    new Add(Graph::get_ptr_of(parent1.get_full_name(), g),
            Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

VecDot &vecdot(const Node &parent1, const Node &parent2, Graph *g,
               const std::string &name) {
  VecDot *node_ptr =
    new VecDot(Graph::get_ptr_of(parent1.get_full_name(), g),
               Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

MatMul &matmul(const Node &parent1, const Node &parent2, Graph *g,
               const std::string &name) {
  MatMul *node_ptr =
    new MatMul(Graph::get_ptr_of(parent1.get_full_name(), g),
               Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

MatSum &matsum(const std::vector<Node *> &parents_ptr, Graph *g,
               const std::string &name) {
  MatSum *node_ptr = new MatSum(parents_ptr, g, name);
  return *node_ptr;
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, Graph *g,
               const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g)},
                g, name);
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               Graph *g, const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g),
                 Graph::get_ptr_of(parent_3.get_full_name(), g)},
                g, name);
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               const Node &parent_4, Graph *g, const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g),
                 Graph::get_ptr_of(parent_3.get_full_name(), g),
                 Graph::get_ptr_of(parent_4.get_full_name(), g)},
                g, name);
}

}
}