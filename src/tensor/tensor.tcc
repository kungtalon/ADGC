#include "tensor/tensor.h"

namespace tensor {

template <typename dType>
Tensor<dType>::Tensor() : Tensor<dType>::Tensor({1}) {}

template <typename dType> Tensor<dType>::Tensor(const TensorShape &shape) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
}

template <typename dType> Tensor<dType>::Tensor(const TensorShape &&shape) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
}

template <typename dType>
Tensor<dType>::Tensor(const TensorShape &shape, const dType &single_value) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_, single_value);
}

template <typename dType>
Tensor<dType>::Tensor(const TensorShape &shape, const dType *values) {
  // dangerous: this constructor does not check whether values has a valid
  // size compatible with the argument shape
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape);
  tensor_ = std::make_shared<std::vector<dType>>(size_);
  memcpy(&(*tensor_->begin()), values, sizeof(dType) * size_);
}

template <typename dType>
Tensor<dType>::Tensor(const TensorShape &shape,
                      const std::vector<dType> &values) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape, values.size());
  tensor_ = std::make_shared<std::vector<dType>>(std::move(values));
}

template <typename dType>
Tensor<dType>::Tensor(const TensorShape &shape,
                      const std::vector<dType> &&values) {
  if (!is_shape_valid(shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  do_shape_update(shape, values.size());
  tensor_ = std::make_shared<std::vector<dType>>(values);
}

template <typename dType>
Tensor<dType>::Tensor(const Tensor<dType> &another)
    : size_(another.size_), dim_(another.dim_), shape_(another.shape_),
      strides_(another.strides_) {
  tensor_ = another.tensor_;
}

template <typename dType>
Tensor<dType>::Tensor(const Tensor<dType> &&another)
    : size_(another.size_), dim_(another.dim_), shape_(another.shape_),
      strides_(another.strides_) {
  tensor_ = another.tensor_;
}

template <typename dType>
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

template <typename dType>
bool Tensor<dType>::operator==(const Tensor<dType> &bt) {
  if (size_ != bt.size_ || shape_ != bt.shape_) {
    return false;
  }
  return tensor_ == bt.tensor_;
}

template <typename dType>
bool Tensor<dType>::operator!=(const Tensor<dType> &bt) {
  return !(*this == bt);
}

template <typename dType>
Tensor<dType> Tensor<dType>::operator[](const size_t &id) {
  if (id >= shape_[0]) {
    throw adg_exception::IndexOutOfRangeError();
  }

  if (shape_[0] == 1) {
    return this->copy();
  }

  TensorShape result_shape;
  if (dim_ == 1) {
    result_shape = {1};
  } else {
    result_shape = TensorShape(shape_.begin() + 1, shape_.end());
  }
  Tensor<dType> result(std::move(result_shape));

  {
    dType *dest_ptr = result.get_tensor_ptr();
    const dType *src_ptr = get_tensor_const_ptr();
    memcpy(dest_ptr, src_ptr + id * strides_[0], sizeof(dType) * strides_[0]);
  }

  return result;
}

template <typename dType>
Tensor<dType> Tensor<dType>::operator[](const TensorSlice &slice) {
  return take(0, slice);
}

template <typename dType>
TensorIterator<dType> Tensor<dType>::get_iterator(const TensorIndex &index) {
  size_t address = 0;
  size_t address_gap = 1; // the address distance between adjacent index
  for (int ix = index.size() - 1; ix >= 0; ix--) {
    address += index[ix] * address_gap;
    address_gap *= shape_[ix];
  }
  return tensor_->begin() + address;
}

template <typename dType>
bool Tensor<dType>::is_shape_valid(const TensorShape &shape) const {
  if (shape == EMPTY_SHAPE) {
    return false;
  }

  if (shape.size() == 0) {
    return false;
  }

  for (size_t length : shape) {
    if (length == 0) {
      return false;
    }
  }

  return true;
}

template <typename dType>
bool Tensor<dType>::is_index_valid(const TensorIndex &index) const {
  if (index.size() == 0) {
    return false;
  }

  if (index.size() != dim_) {
    return false;
  }

  for (int ix = 0; ix < index.size(); ++ix) {
    if (index[ix] >= shape_[ix]) {
      return false;
    }
  }

  return true;
}

template <typename dType>
void Tensor<dType>::set_value(const TensorIndex &index, const dType &value) {
  if (!is_index_valid(index)) {
    throw adg_exception::InvalidTensorIndexException();
  }

  TensorIterator<dType> iter = get_iterator(index);
  *iter = value;
}

// when tensor has only one element
template <typename dType> dType Tensor<dType>::get_value() const {
  if (size_ != 1) {
    throw adg_exception::InvalidTensorIndexException();
  }

  return *tensor_->begin();
}

template <typename dType>
dType Tensor<dType>::get_value(const TensorIndex &index) {
  if (!is_index_valid(index)) {
    throw adg_exception::InvalidTensorIndexException();
  }

  return *get_iterator(index);
}

template <typename dType> Tensor<int32_t> Tensor<dType>::to_int() const {
  std::vector<int32_t> values(tensor_->begin(), tensor_->end());
  return Tensor<int32_t>(shape_, std::move(values));
}

template <typename dType> Tensor<float> Tensor<dType>::to_float() const {
  std::vector<float> values(tensor_->begin(), tensor_->end());
  return Tensor<float>(shape_, std::move(values));
}

template <typename dType> Tensor<double> Tensor<dType>::to_double() const {
  std::vector<double> values(tensor_->begin(), tensor_->end());
  return Tensor<double>(shape_, std::move(values));
}

/*
take slices the tensor along a specific axis
A simpler version :
  for (int i =0; i < size / strides[axis-1]; ++i) {
    start = i * strides[axis-1]
    for (int j : slice) {
        s = start + j * strides[axis]
        memcpy(dest, s, s + strides[axis])
    }
  }
*/
template <typename dType>
Tensor<dType> Tensor<dType>::take(const size_t &axis,
                                  const TensorSlice &slice) {
  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError();
  }

  for (size_t i : slice) {
    if (i >= shape_[axis]) {
      throw adg_exception::IndexOutOfRangeError();
    }
  }

  size_t slice_len = slice.size();
  TensorShape result_shape = shape_;
  result_shape[axis] = slice_len;
  Tensor<dType> result(std::move(result_shape));

  std::vector<size_t> indices = slice;
  for (size_t &index : indices) {
    index *= strides_[axis];
  }

  dType *dest_ptr = result.get_tensor_ptr();
  const dType *src_ptr = get_tensor_const_ptr();
  size_t max_iter = (axis == 0) ? 1 : size_ / strides_[axis - 1];
  for (int ix = 0; ix < max_iter; ++ix) {
    for (size_t &index : indices) {
      if (axis != dim_ - 1) {
        memcpy(dest_ptr, src_ptr + index, sizeof(dType) * strides_[axis]);
      } else {
        *dest_ptr = *(src_ptr + index);
      }
      dest_ptr += strides_[axis];
      index += strides_[axis - 1];
    }
  }

  return result;
}

// get_dot_shape returns the shape of result matrix for dot_mul
template <typename dType>
TensorShape Tensor<dType>::get_dot_shape(const Tensor<dType> &bt) const {
  if (get_shape() == EMPTY_SHAPE || bt.get_shape() == EMPTY_SHAPE) {
    throw adg_exception::EmptyTensorError();
  }

  // mat mul will only change the last two dimensions
  // first check the other dimension are aligned
  if (get_dim() != bt.get_dim()) {
    return EMPTY_SHAPE;
  }

  size_t cur_dim = get_dim();
  TensorShape shape_left = get_shape();
  TensorShape shape_right = bt.get_shape();

  for (int ix = 0; ix < cur_dim - 2; ix++) {
    if (shape_left[ix] != shape_right[ix]) {
      return EMPTY_SHAPE;
    }
  }

  // [a, b], [b, c] -> [a, c]
  // check whether left.shape[-1] == right.shape[-2]
  if (shape_left[cur_dim - 1] != shape_right[cur_dim - 2]) {
    throw adg_exception::MismatchTensorShapeError();
  }

  // then, the answer is [..., a, c]
  TensorShape result_shape(shape_left.begin(), shape_left.end() - 1);
  result_shape.push_back(shape_right[cur_dim - 1]);
  return result_shape;
}

// dot implements the matrix multiplication
template <typename dType>
Tensor<dType> Tensor<dType>::dot(const Tensor<dType> &bt) const {
  TensorShape result_shape = get_dot_shape(bt);
  Tensor<dType> result(result_shape);

  size_t M = shape_[dim_ - 2];
  size_t N = bt.shape_[dim_ - 1];
  size_t K = shape_[dim_ - 1];

  utils::math::tensor_gemm(size_, bt.size_, result.size_, M, N, K,
                           get_tensor_const_ptr(), bt.get_tensor_const_ptr(),
                           result.get_tensor_ptr());
  return result;
}

// multiply implements the element-wise multiplication
template <typename dType>
Tensor<dType> Tensor<dType>::multiply(const Tensor<dType> &bt) const {
  if (bt.shape_ != shape_) {
    throw adg_exception::MismatchTensorShapeError();
  }

  Tensor<dType> result = Tensor(shape_);
  utils::math::elementwise_multiply(size_, get_tensor_const_ptr(),
                                    bt.get_tensor_const_ptr(),
                                    result.get_tensor_ptr());
  return result;
}

// multiply implements the element-wise multiplication
template <typename dType>
Tensor<dType> Tensor<dType>::multiply(const double &multiplier) const {

  Tensor<dType> result = Tensor(shape_, static_cast<dType>(multiplier));
  utils::math::elementwise_multiply(size_, get_tensor_const_ptr(),
                                    result.get_tensor_const_ptr(),
                                    result.get_tensor_ptr());
  return result;
}

template <typename dType>
Tensor<dType> Tensor<dType>::add(const Tensor<dType> &bt) const {
  if (bt.shape_ != shape_) {
    throw adg_exception::MismatchTensorShapeError();
  }

  Tensor<dType> result = Tensor(shape_);
  utils::math::elementwise_add(size_, get_tensor_const_ptr(),
                               bt.get_tensor_const_ptr(),
                               result.get_tensor_ptr());
  return result;
}

template <typename dType>
Tensor<dType> Tensor<dType>::add(const double &number) const {
  Tensor<dType> result = Tensor(shape_, static_cast<dType>(number));
  utils::math::elementwise_add(size_, get_tensor_const_ptr(),
                               result.get_tensor_const_ptr(),
                               result.get_tensor_ptr());
  return result;
}

template <typename dType>
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

// copy returns a deep copy of current tensor
template <typename dType> Tensor<dType> Tensor<dType>::copy() const {
  return Tensor<dType>(shape_, get_tensor_const_ptr());
}

template <typename dType>
Tensor<dType> Tensor<dType>::sum(const size_t &axis) const {
  if (axis == SIZE_MAX) {
    dType res = utils::math::sum(size_, get_tensor_const_ptr(), 1);
    return Tensor<dType>({1}, res);
  }

  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError();
  }

  // only sum along the given axis
  TensorShape result_shape = shape_;
  result_shape.erase(result_shape.begin() + axis);
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

template <typename dType>
void Tensor<dType>::do_shape_update(const TensorShape &shape,
                                    const size_t &keep_size) {
  size_t tmp_size = 1;
  size_t tmp_dim = shape.size();
  TensorShape tmp_strides = TensorShape(tmp_dim, 1);

  for (int ix = tmp_dim - 1; ix >= 0; --ix) {
    tmp_strides[ix] = tmp_size;
    tmp_size *= shape[ix];
  }

  if (keep_size) {
    if (tmp_size != keep_size) {
      throw adg_exception::InvalidTensorShapeException();
    }
  }

  size_ = tmp_size;
  strides_ = tmp_strides;
  shape_ = shape;
  dim_ = shape.size();
}

// reshape changes the shape_, dim_, strides_ of the tensor
template <typename dType>
void Tensor<dType>::reshape(const TensorShape &new_shape) {
  if (!is_shape_valid(new_shape)) {
    throw adg_exception::InvalidTensorShapeException();
  }

  // keep same size
  do_shape_update(new_shape, size_);
}

// helper function for transpose, computes the coordinate of an element on a
// specific axis given its vector index
template <typename dType>
size_t Tensor<dType>::get_coordinate_at_axis(const size_t &ind,
                                             const size_t &axis,
                                             const TensorShape &strides) {
  // we convert the array index to multi-array coordinate by
  // taking modulo with regard to the strides_
  if (axis == 0) {
    return ind / strides[0];
  }
  return (ind % strides[axis - 1]) / strides[axis];
}

/*
  This is helper function for transpose, computes the new index after transpose.
  Suppose c is the vector of coordinates and s is the vector of strides.
  The conversion equation for multi-array coordinate and the flattened index is
    ` ind = sum_i {c[i] * s[i]} `
  After transpose, we should have the new strides ns and only switch the
  coordinate of axis a and b, then
    ` new_ind = sum_{i != a, b} {c[i] * ns[i]} + c[a] * ns[b] + c[b] * ns[a] `
  The `offset` computed in this function is new_ind - ind
*/
template <typename dType>
size_t Tensor<dType>::get_index_after_transpose(
    const size_t &ind, const size_t &axis_a, const size_t &axis_b,
    const TensorShape &ori_strides, const TensorShape &new_strides) {
  int64_t signed_index = ind;
  int64_t offset = 0;
  {
    int64_t coord_a = get_coordinate_at_axis(ind, axis_a, ori_strides);
    int64_t coord_b = get_coordinate_at_axis(ind, axis_b, ori_strides);
    offset += coord_a * (new_strides[axis_b] - ori_strides[axis_a]) +
              coord_b * (new_strides[axis_a] - ori_strides[axis_b]);
  }

  {
    int64_t cur_coord;
    for (size_t ax = axis_a + 1; ax < axis_b; ++ax) {
      cur_coord = get_coordinate_at_axis(ind, ax, ori_strides);
      offset += cur_coord * (new_strides[ax] - ori_strides[ax]);
    }
  }

  int64_t new_index = signed_index + offset;
  return static_cast<size_t>(new_index);
}

// do_transpose impl of transpose
// maps the index of original tensor to the index of the new tensor
template <typename dType>
void Tensor<dType>::do_transpose(const size_t &axis_a, const size_t &axis_b,
                                 Tensor<dType> &dest_tensor) const {

  auto get_new_index = [&](const size_t &index) {
    return get_index_after_transpose(index, axis_a, axis_b, this->strides_,
                                     dest_tensor.strides_);
  };

  TensorIterator<dType> src_iter = tensor_->begin();
  TensorIterator<dType> dest_iter = dest_tensor.tensor_->begin();
  size_t src_index = 0;
  size_t dest_index;
  // iterate all elements in tensor
  while (src_iter != tensor_->end()) {
    dest_index = get_new_index(src_index++);
    *(dest_iter + dest_index) = *(src_iter++);
  }
}

// transpose will switch the dimension of two axes
template <typename dType>
Tensor<dType> Tensor<dType>::transpose(const size_t &axis_ai,
                                       const size_t &axis_bi) const {
  if (axis_ai == axis_bi) {
    throw std::invalid_argument("Repeated axis in transpose");
  }

  size_t axis_a = std::min(axis_ai, axis_bi);
  size_t axis_b = std::max(axis_ai, axis_bi);

  if (axis_b >= dim_) {
    throw adg_exception::AxisOutOfRangeError();
  }

  // deep copy
  Tensor<dType> result = this->copy();

  // update the shape_ and strides_
  std::iter_swap(result.shape_.begin() + axis_a,
                 result.shape_.begin() + axis_b);

  size_t new_stride = 1;
  for (int ix = dim_ - 1; ix >= 0; --ix) {
    result.strides_[ix] = new_stride;
    new_stride *= result.shape_[ix];
  }

  do_transpose(axis_a, axis_b, result);

  return result;
}

// by default, we transpose the last two axes for convenient matrix operation
template <typename dType> Tensor<dType> Tensor<dType>::transpose() const {
  if (dim_ <= 1) {
    throw adg_exception::AxisOutOfRangeError();
  }

  return transpose(dim_ - 2, dim_ - 1);
}

// short for transpose
template <typename dType> Tensor<dType> Tensor<dType>::t() const {
  if (dim_ <= 1) {
    throw adg_exception::AxisOutOfRangeError();
  }

  return transpose(dim_ - 2, dim_ - 1);
}

template <typename dType>
void Tensor<dType>::fill_diag(const std::vector<dType> &diag_values) {
  if (dim_ != 2) {
    throw adg_exception::InvalidTensorShapeException();
  }
  size_t diag_len = diag_values.size();
  size_t tensor_min_dim_len = std::min(shape_[0], shape_[1]);

  if (diag_len > tensor_min_dim_len) {
    throw adg_exception::MismatchTensorShapeError();
  }

  utils::math::fill_diagonal(std::min(shape_[0], diag_len), shape_[1],
                             &*diag_values.begin(), get_tensor_ptr());
}

template <typename dType> void Tensor<dType>::map(Mapper<dType> &mapper) {
#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  throw adg_exception::TestingDebugException("Not here");
  mapper.run(get_tensor_ptr(), size_);
#endif
}

template <typename dType> void Tensor<dType>::map(Mapper<dType> &&mapper) {
#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  throw adg_exception::TestingDebugException("Not here");
  mapper.run(get_tensor_ptr(), size_);
#endif
}

template <typename dType>
void Tensor<dType>::map(const std::function<void(dType &)> &func) {
  Mapper<dType> mapper(func);

#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  throw adg_exception::TestingDebugException("Not here");
  mapper.run(get_tensor_ptr(), size_);
#endif
}

// // instantiation
// template class Tensor<double>;
// template class Tensor<int32_t>;
// template class Tensor<float>;

// TensorIterator<double> inst_tensor_iter_double;
// TensorIterator<int32_t> inst_tensor_iter_int;
// TensorIterator<float> inst_tensor_iter_float;

} // namespace tensor