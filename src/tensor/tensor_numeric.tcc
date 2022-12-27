//
// All implementations about tensor's numeric computations like matrix multiplication, summation
//

#include "tensor/tensor.h"

namespace tensor {

template<typename dType>
Tensor<dType> Tensor<dType>::operator-() const {
  Tensor<dType> result = this->copy();
  utils::math::elementwise_negative(result.size_, result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> &Tensor<dType>::operator+=(const Tensor<dType> &bt) {
  // in-place addition
  if (bt.get_shape() != shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> operator-=: get mismatched shapes: " +
        utils::vector_to_str(bt.get_shape()) + " and " +
        utils::vector_to_str(shape_));
  }

  utils::math::elementwise_add_inplace(size_, get_tensor_ptr(),
                                       bt.get_tensor_const_ptr());
  return *this;
}

template<typename dType>
Tensor<dType> &Tensor<dType>::operator+=(const dType &number) {
  // in-place addition
  utils::math::elementwise_addn(size_, get_tensor_ptr(), number);
  return *this;
}

template<typename dType>
Tensor<dType> &Tensor<dType>::operator-=(const Tensor<dType> &bt) {
  // in-place subtraction
  if (bt.get_shape() != shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> operator-=: get mismatched shapes: " +
        utils::vector_to_str(bt.get_shape()) + " and " +
        utils::vector_to_str(shape_));
  }

  utils::math::elementwise_add_inplace(size_, get_tensor_ptr(),
                                       bt.get_tensor_const_ptr(), true);
  return *this;
}

template<typename dType>
Tensor<dType> &Tensor<dType>::operator-=(const dType &number) {
  // in-place subtraction
  utils::math::elementwise_addn(size_, get_tensor_ptr(), number, true);
  return *this;
}

// get_dot_shape returns the shape of result matrix for dot_mul
template<typename dType>
TensorShape Tensor<dType>::get_dot_shape(const Tensor<dType> &bt) const {
  if (get_shape() == EMPTY_SHAPE || bt.get_shape() == EMPTY_SHAPE) {
    throw adg_exception::EmptyTensorError();
  }

  bool tensor_dot_matrix = false;
  // if tensor dot tensor, mat mul will only change the last two dimensions
  // if tensor dot matrix...
  if (get_dim() != bt.get_dim()) {
    if (bt.get_dim() == 2 || get_dim() == 2) {
      tensor_dot_matrix = true;
    } else {
      throw adg_exception::MismatchTensorDimError(
        "Tensor >> get_dot_shape: the tensor should have either dim 2, or the same dim as the other");
    }
  }

  size_t left_dim = get_dim();
  size_t right_dim = bt.get_dim();
  TensorShape shape_left, shape_right, shape_larger;
  // make the left shape with the bigger dim
  if (left_dim < right_dim) {
    shape_larger = bt.get_shape();
  } else {
    shape_larger = get_shape();
  }

  shape_left = get_shape();
  shape_right = bt.get_shape();

  if (!tensor_dot_matrix) {
    for (int ix = 0; ix < left_dim - 2; ix++) {
      if (shape_left[ix] != shape_right[ix]) {
        throw adg_exception::MismatchTensorDimError(
          "Tensor >> get_dot_shape: MismatchTensorDimError.");
      }
    }
  }

  // [a, b], [b, c] -> [a, c]
  // check whether left.shape[-1] == right.shape[-2]
  if (shape_left[left_dim - 1] != shape_right[right_dim - 2]) {
    throw adg_exception::MismatchTensorShapeError(
      "MismatchTensorShapeError >> get_dot_shape\n Left shape is " +
        utils::array_to_str(1, dim_, &*shape_left.begin()) + "Right shape is " +
        utils::array_to_str(1, bt.get_dim(), &*shape_right.begin()));
  }

  // then, the answer is [..., a, c]
  TensorShape result_shape(shape_larger.begin(), shape_larger.end() - 2);
  result_shape.emplace_back(shape_left[left_dim - 2]);
  result_shape.emplace_back(shape_right[right_dim - 1]);
  return result_shape;
}

// dot implements the matrix multiplication
template<typename dType>
Tensor<dType> Tensor<dType>::dot(const Tensor<dType> &bt) const {
  TensorShape result_shape = get_dot_shape(bt);
  Tensor<dType> result(result_shape);

  size_t M = shape_[dim_ - 2];
  size_t N = bt.shape_[bt.get_dim() - 1];
  size_t K = shape_[dim_ - 1];

  utils::math::tensor_gemm(size_, bt.size_, result.size_, M, N, K,
                           get_tensor_const_ptr(), bt.get_tensor_const_ptr(),
                           result.get_tensor_ptr());
  return result;
}

// multiply implements the element-wise multiplication
template<typename dType>
Tensor<dType> Tensor<dType>::multiply(const Tensor<dType> &bt) const {
  if (bt.shape_ != shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "MismatchTensorShapeError >> multiply " + utils::vector_to_str(shape_) +
        " and " + utils::vector_to_str(bt.shape_));
  }

  Tensor<dType> result = Tensor(shape_);
  utils::math::elementwise_multiply(size_, get_tensor_const_ptr(),
                                    bt.get_tensor_const_ptr(),
                                    result.get_tensor_ptr());
  return result;
}

// multiply implements the element-wise multiplication
template<typename dType>
Tensor<dType> Tensor<dType>::multiply(const dType &multiplier) const {
  Tensor<dType> result = Tensor(shape_, multiplier);
  utils::math::elementwise_multiply(size_, get_tensor_const_ptr(),
                                    result.get_tensor_const_ptr(),
                                    result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::div(const dType &denom) const {
  Tensor<dType> result = Tensor(shape_, denom);
  utils::math::elementwise_divide(size_, get_tensor_const_ptr(),
                                  result.get_tensor_const_ptr(),
                                  result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::div(const Tensor<dType> &bt) const {
  if (shape_ != bt.shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> div: MismatchTensorShapeError: get shape " +
        utils::vector_to_str(shape_) + " and " +
        utils::vector_to_str(bt.shape_));
  }

  Tensor<dType> result(shape_);
  utils::math::elementwise_divide(size_, get_tensor_const_ptr(), bt.get_tensor_const_ptr(), result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::add(const Tensor<dType> &bt) const {
  if (bt.shape_ != shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "MismatchTensorShapeError >> add " + utils::vector_to_str(shape_) +
        " and " + utils::vector_to_str(bt.shape_));
  }

  Tensor<dType> result = Tensor(std::move(shape_));
  utils::math::elementwise_add(size_, get_tensor_const_ptr(),
                               bt.get_tensor_const_ptr(),
                               result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::add(const dType &number) const {
  Tensor<dType> result = Tensor(shape_, static_cast<dType>(number));
  utils::math::elementwise_add(size_, get_tensor_const_ptr(),
                               result.get_tensor_const_ptr(),
                               result.get_tensor_ptr());
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::sub(const Tensor<dType> &bt) const {
  if (bt.shape_ != shape_) {
    throw adg_exception::MismatchTensorShapeError(
      "MismatchTensorShapeError >> sub " + utils::vector_to_str(shape_) +
        " and " + utils::vector_to_str(bt.shape_));
  }

  Tensor<dType> result = Tensor(std::move(shape_));
  utils::math::elementwise_add(size_, get_tensor_const_ptr(),
                               bt.get_tensor_const_ptr(),
                               result.get_tensor_ptr(), true);
  return result;
}

// matrix kronecker product
template<typename dType>
Tensor<dType> Tensor<dType>::kron(const Tensor<dType> &lt,
                                  const Tensor<dType> &rt) {
  if (lt.dim_ < 2 || rt.dim_ < 2) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> kron: not enough dimension");
  }

  if (lt.dim_ != rt.dim_) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> kron: MismatchTensorShapeError");
  }

  for (size_t ix = 0; ix < lt.dim_ - 2; ++ix) {
    if (lt.shape_[ix] != rt.shape_[ix]) {
      throw adg_exception::MismatchTensorShapeError(
        "Tensor >> kron: MismatchTensorShapeError at axis " + std::to_string(ix));
    }
  }

  size_t left_nrow = lt.shape_[lt.dim_ - 2];
  size_t right_nrow = rt.shape_[rt.dim_ - 2];
  size_t left_ncol = lt.shape_[lt.dim_ - 1];
  size_t right_ncol = rt.shape_[rt.dim_ - 1];

  TensorShape result_shape = lt.shape_;
  result_shape[lt.dim_ - 2] = left_nrow * right_nrow;
  result_shape[lt.dim_ - 1] = left_ncol * right_ncol;
  Tensor<dType> result(std::move(result_shape));

  utils::math::tensor_kron_product(
    lt.size_, rt.size_, left_nrow, left_ncol, right_nrow, right_ncol,
    lt.get_tensor_const_ptr(), rt.get_tensor_const_ptr(),
    result.get_tensor_ptr());

  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::concat(const std::vector<Tensor<dType>> &tensors, const size_t &axis) {
  if (tensors.size() == 1) {
    return tensors[0].copy();
  }

  TensorShape src_shape = tensors[0].get_shape();
  size_t len_after_concat = 0;

  // check shape: except the axis to be stacked together, dimension should keep the same through all tensors
  bool check_shape = true;
  for (auto ts : tensors) {
    TensorShape cur_shape = ts.get_shape();
    len_after_concat += cur_shape[axis];
    if (ts.get_shape() == src_shape) {
      continue;
    }
    if (ts.get_dim() != src_shape.size()) {
      check_shape = false;
      break;
    }
    for (size_t ax = 0; ax < src_shape.size(); ++ax) {
      if (ax != axis && cur_shape[ax] != src_shape[ax]) {
        check_shape = false;
        break;
      }
    }
    if (!check_shape) { break; }
  }
  if (!check_shape) {
    throw adg_exception::MismatchTensorShapeError("Tensor >> concat: MismatchTensorShapeError");
  }

  TensorShape target_shape = src_shape;
  target_shape[axis] = len_after_concat;
  Tensor<dType> result(std::move(target_shape));

  // tensor_offset: the coordinate of the leftmost element of the current tensor on the axis after concat
  size_t tensor_offset = 0, new_index;
  dType *result_tensor_ptr = result.get_tensor_ptr();
  TensorShape result_strides = result.get_strides();
  for (size_t ix = 0; ix < tensors.size(); ++ix) {
    const dType *src_tensor_ptr = tensors[ix].get_tensor_const_ptr();
    TensorShape cur_strides = tensors[ix].get_strides();

    for (size_t index = 0; index < tensors[ix].get_size(); index += cur_strides[axis]) {
      new_index = get_index_after_concat(index, axis, tensor_offset,
                                         cur_strides, result_strides);
      memcpy(result_tensor_ptr + new_index, src_tensor_ptr + index, sizeof(dType) * cur_strides[axis]);
    }

    tensor_offset += tensors[ix].get_shape()[axis];
  }
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::mean(const size_t &axis, bool keep_dim) const {
  Tensor<dType> result = sum(axis, keep_dim);
  size_t len_at_axis = shape_[axis];
  result.map([&len_at_axis](dType &val) { val /= len_at_axis; });
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::max(const size_t &axis, bool keep_dim) const {
  if (axis == SIZE_MAX) {
    return Tensor<dType>({1}, *std::max_element(tensor_->begin(), tensor_->end()));
  }

  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> sum: axis out of range");
  }

  // only take max along the given axis
  TensorShape result_shape = shape_;
  if (!keep_dim) {
    result_shape.erase(result_shape.begin() + axis);
  } else {
    result_shape[axis] = 1;
  }
  Tensor<dType> result(std::move(result_shape));

  auto dest_ptr = result.get_tensor_ptr();
  auto src_ptr = get_tensor_const_ptr();
  size_t src_index = 0;
  while (src_index < size_) {
    if (get_coordinate_at_axis(src_index, axis, strides_) == 0) {
      *(dest_ptr++) =
        utils::math::max(shape_[axis], src_ptr + src_index, strides_[axis]);
    }
    ++src_index;
  }
  return result;
}

template<typename dType>
Tensor<dType> Tensor<dType>::arg_amax(const size_t &axis, bool keep_dim) const {
  if (axis == SIZE_MAX) {
    return Tensor<dType>({1}, static_cast<dType>(utils::math::arg_amax(size_, get_tensor_const_ptr(), 1)));
  }

  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> sum: axis out of range");
  }

  // only take max along the given axis
  TensorShape result_shape = shape_;
  if (!keep_dim) {
    result_shape.erase(result_shape.begin() + axis);
  } else {
    result_shape[axis] = 1;
  }
  Tensor<dType> result(std::move(result_shape));

  auto dest_ptr = result.get_tensor_ptr();
  auto src_ptr = get_tensor_const_ptr();
  size_t src_index = 0;
  while (src_index < size_) {
    if (get_coordinate_at_axis(src_index, axis, strides_) == 0) {
      *(dest_ptr++) =
        utils::math::arg_amax(shape_[axis], src_ptr + src_index, strides_[axis]);
    }
    ++src_index;
  }
  return result;
}

template<typename dType>
void Tensor<dType>::normal_init(double loc, double scale, size_t seed) {
  size_t r_seed = seed;
  if (r_seed == SIZE_MAX) {
    r_seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  std::default_random_engine eng(r_seed);
  std::normal_distribution<double> distribution(loc, scale);

  for (auto it = tensor_->begin(); it != tensor_->end(); it++) {
    double number = distribution(eng);
    *it = static_cast<dType>(number);
  }
}

template<typename dType>
Tensor<dType> Tensor<dType>::sum(const size_t &axis, bool keep_dim) const {
  if (axis == SIZE_MAX) {
    dType res = utils::math::sum(size_, get_tensor_const_ptr(), 1);
    return Tensor<dType>({1}, res);
  }

  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> sum: axis out of range");
  }

  // only sum along the given axis
  TensorShape result_shape = shape_;
  if (!keep_dim) {
    result_shape.erase(result_shape.begin() + axis);
  } else {
    result_shape[axis] = 1;
  }
  Tensor<dType> result(std::move(result_shape));

  auto dest_ptr = result.get_tensor_ptr();
  auto src_ptr = get_tensor_const_ptr();
  size_t src_index = 0;
  while (src_index < size_) {
    if (get_coordinate_at_axis(src_index, axis, strides_) == 0) {
      *(dest_ptr++) =
        utils::math::sum(shape_[axis], src_ptr + src_index, strides_[axis]);
    }
    ++src_index;
  }
  return result;
}

}