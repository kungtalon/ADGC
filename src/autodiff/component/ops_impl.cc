#include "autodiff/component/ops.h"

namespace auto_diff {
namespace ops {

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

Pad2D::Pad2D(Node *parent_ptr, const std::vector<std::pair<size_t, size_t>> &padding, const double &value,
             Graph *g, const std::string &name)
  : Node(NodeType::ADG_PAD2D_TYPE, {parent_ptr}, name, g), pad_value_(value), padding_(padding) {
  set_backward_version(1);
  if (padding.size() != 2) {
    throw adg_exception::InvalidNodeArgumentError(
      "Pad2D >> Pad2D: expect padding for 2 dimensions, got " + std::to_string(padding.size()));
  }
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

Conv2D::Conv2D(Node *input_ptr,
               Parameter *kernel_ptr,
               const std::vector<size_t> &strides,
               Graph *g,
               const std::string &name)
  : Node(NodeType::ADG_CONV2D_TYPE, {input_ptr, kernel_ptr}, name, g),
    strides_(strides) {
  set_backward_version(1);
  // second being the kernel
  // input : image features [B, H, W, Cin], kernel [Kh, Kw, Cin, Cout]
  // if input.size() == 3: use bias of size : [Cout]
  if (parents_.size() != 2) {
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
  size_t h = image_shape[1];
  size_t w = image_shape[2];
  size_t in_c = image_shape[3];
  if (kernel_shape_[2] != in_c) {
    throw adg_exception::MismatchNodeValueShapeError(
      "Conv >> Conv: different channel size for image and kernel! "
        + std::to_string(kernel_shape_[2]) + " and " + std::to_string(in_c));
  }

  out_c_ = parents_[1]->get_value_shape()[parents_[1]->get_value_dim() - 1];
  out_h_ = (h - kernel_shape_[0]) / strides[0] + 1;
  out_w_ = (w - kernel_shape_[1]) / strides[1] + 1;

  value_ = DTensor({out_c_, out_h_, out_w_});
}

void Conv2D::do_forward() {
  col_kernel_ = parents_[1]->get_value().copy();
  col_kernel_.reshape({kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2], out_c_});
  // shape: [kh * kw * cin, cout]

  im2col(parents_[1]->get_value()); // shape: [B, (h - kh) * (w - kw), kh * kw * cin]

  value_ = col_image_.dot(col_kernel_); // shape: [B, (h-kh)*(w-kw), c_out]
  value_.reshape({out_h_, out_w_, out_c_});
}

DTensor Conv2D::do_backward(Node *parent_ptr) {

}

void Conv2D::im2col(const DTensor &input) {
  // input : [b, h, w, c]
  size_t row_steps, col_steps;
  tensor::TensorShape shape = input.get_shape();
  row_steps = shape[1] / strides_[0];
  col_steps = shape[2] / strides_[1];

  col_image_ = DTensor({shape[0], out_h_ * out_w_, shape[3]});
  size_t window_size = kernel_shape_[0] * kernel_shape_[1];

  size_t cur_window_index;   // index of the windows' leftmost element
  for (size_t ib = 0; ib < shape[0]; ++ib) {
    for (size_t ih = 0; ih < shape[1]; ih += strides_[0]) {
      for (size_t iw = 0; iw < shape[2]; iw += strides_[1]) {
        for (size_t ii = 0; ii < window_size; ++ii) {

        }
      }
    }
  }
}

} // namespace ops
} // namespace auto_diff
