#include "autodiff/component/ops.h"

namespace auto_diff {
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
  if (parents_[1]->get_value_size() == 1) {
    value_ =
      parents_[0]->get_value().add(parents_[1]->get_value().get_value({0}));
    return;
  }
  value_ = parents_[0]->get_value().add(parents_[1]->get_value());
}

DTensor Add::do_backward(Node *parent_ptr) {
  if (parents_[1]->get_value_size() == 1 && parent_ptr == parents_[1]) {
    return tensor::Ones({1, get_value_size()});
  }
  return tensor::Eye(get_value_size());
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
  value_ = tensor::dot(l_mat, parents_[1]->get_value());
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

  // aggregate the batch dim with the row dim... it is equivalent
  DTensor left = parents_[0]->get_value();
  size_t left_last_len = left.get_shape()[left.get_dim() - 1];
  left.reshape({left.get_size() / left_last_len, left_last_len});
  if (parent_ptr == parents_[0]) {
    // C = A * B
    // jacobi(C, A) = kron(I, B)
    tensor::TensorShape left_shape = parents_[0]->get_value_shape();
    size_t left_row = left.get_shape()[0];
    return DTensor::kron(tensor::Eye(left_row), parents_[1]->get_value());
  } else {
    // jacobi(C, B) = kron(A.T, I)
    tensor::TensorShape right_shape = parents_[1]->get_value_shape();
    size_t right_col = right_shape[right_shape.size() - 1];
    return DTensor::kron(left.t(), tensor::Eye(right_col));
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

Reshape::Reshape(Node *parent_ptr, const tensor::TensorShape &shape, Graph *g,
                 const std::string &name)
  : Node(NodeType::ADG_RESHAPE_TYPE, {parent_ptr}, name, g) {

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
  return tensor::Eye(get_value_size());
}

PointMul::PointMul(Node *parent_ptr1, Node *parent_ptr2, Graph *g, const std::string &name)
  : Node(NodeType::ADG_POINTMUL_TYPE, {parent_ptr1, parent_ptr2}, name, g) {
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
    return parents_[1]->get_value();
  } else {
    return parents_[0]->get_value();
  }
}

Conv::Conv(Node *input_ptr,
           Parameter *kernel_ptr,
           const std::vector<size_t> &strides,
           Graph *g,
           const std::string &name) : Node(NodeType::ADG_CONV_TYPE, {input_ptr, kernel_ptr}, name, g),
                                      strides_(strides) {
  // second being the kernel
  // input : image features [B, H, W, Cin], kernel [Kh, Kw, Cin, Cout]
  // if input.size() == 3: use bias of size : [Cout]
  if (parents_.size() < 2 || parents_.size() > 3) {
    throw adg_exception::OpsParentsNumException("Conv >> Conv: expect 2 parent nodes...");
  }

  if (strides_.size() != 2) {
    throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: strides should have size of 2...");
  }

  for (size_t &stride : strides_) {
    if (stride < 1) {
      throw adg_exception::InvalidNodeArgumentError("Conv >> Conv: getting stride smaller than 1..");
    }
  }

  auto image_shape = parents_[0]->get_value_shape();
  kernel_shape_ = parents_[1]->get_value_shape();
  size_t h = image_shape[0];
  size_t w = image_shape[1];
  size_t in_c = image_shape[2];
  size_t kh = kernel_shape_[0];
  size_t kw = kernel_shape_[1];
  if (kernel_shape_[2] != in_c) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Conv >> Conv: different channel size for image and kernel! "
        + std::to_string(kernel_shape_[2]) + " and " + std::to_string(in_c));
  }

  out_c_ = parents_[1]->get_value_shape()[parents_[1]->get_value_dim() - 1];
  out_h_ = (h - kh) / strides[0] + 1;
  out_w_ = (w - kw) / strides[1] + 1;

  value_ = DTensor({out_c_, out_h_, out_w_});
}

void Conv::do_forward() {
  col_kernel_ = parents_[1]->get_value().copy();
  col_kernel_.reshape({kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2], out_c_});
  // shape: [kh * kw * cin, cout]

  col_image_ = im2col(parents_[1]->get_value()); // shape: [B, (h - kh) * (w - kw), kh * kw * cin]

  value_ = col_image_.dot(col_kernel_); // shape: [B, (h-kh)*(w-kw), c_out]
  value_.reshape({out_h_, out_w_, out_c_});
}

DTensor Conv::do_backward(Node *parent_ptr) {

}

DTensor Conv::im2col(const DTensor &input) {

}

} // namespace ops
} // namespace auto_diff
