#include "tensor/tensor.h"

namespace tensor {

template<typename dType>
Tensor<dType>::Tensor() : Tensor<dType>::Tensor({1}) {}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &shape) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &&shape) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &shape, const dType &single_value) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_, single_value);
}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &shape, const dType *values) {
  // dangerous: this constructor does not check whether values has a valid
  // size compatible with the argument shape
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
  memcpy(&(*tensor_->begin()), values, sizeof(dType) * size_);
}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &shape,
                      const std::vector<dType> &values) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape, values.size());
  tensor_ = std::make_shared<std::vector<dType>>(std::move(values));
}

template<typename dType>
Tensor<dType>::Tensor(const TensorShape &shape,
                      const std::vector<dType> &&values) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException(
      "InvalidTensorShapeException; Failed when constructing: " + utils::vector_to_str(shape));
  }

  do_shape_update(shape, values.size());
  tensor_ = std::make_shared<std::vector<dType>>(values);
}

template<typename dType>
Tensor<dType>::Tensor(const Tensor<dType> &another)
  : size_(another.size_), dim_(another.dim_), shape_(another.shape_),
    strides_(another.strides_) {
  tensor_ = another.tensor_;
}

template<typename dType>
Tensor<dType>::Tensor(const Tensor<dType> &&another)
  : size_(another.size_), dim_(another.dim_), shape_(another.shape_),
    strides_(another.strides_) {
  tensor_ = another.tensor_;
}

template<typename dType>
Tensor<dType> &Tensor<dType>::operator=(const Tensor<dType> &bt) {
  if (tensor_ == bt.tensor_) {
    // nothing to do
    return *this;
  }

  shape_ = bt.shape_;
  dim_ = bt.dim_;
  strides_ = bt.strides_;
  size_ = bt.size_;
  tensor_ = bt.tensor_;
  return *this;
}

template<typename dType>
bool Tensor<dType>::operator==(const Tensor<dType> &bt) const {
  if (size_ != bt.size_ || shape_ != bt.shape_) {
    return false;
  }
  return tensor_ == bt.tensor_;
}

template<typename dType>
bool Tensor<dType>::operator!=(const Tensor<dType> &bt) const {
  return !(*this == bt);
}


// // instantiation
// template class Tensor<double>;
// template class Tensor<int32_t>;
// template class Tensor<float>;

// TensorIterator<double> inst_tensor_iter_double;
// TensorIterator<int32_t> inst_tensor_iter_int;
// TensorIterator<float> inst_tensor_iter_float;

} // namespace tensor