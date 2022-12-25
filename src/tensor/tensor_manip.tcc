//
// All implementations about tensor's manipulation, like reshape, transpose
//

#include "tensor/tensor.h"

namespace tensor {

template<typename dType>
Tensor<dType> Tensor<dType>::operator[](const size_t &id) const {
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

template<typename dType>
Tensor<dType> Tensor<dType>::operator[](const std::vector<size_t> &slice_indice) const {
  return take(0, slice_indice);
}

template<typename dType>
TensorIterator<dType> Tensor<dType>::get_iterator(const TensorIndex &index) {
  size_t address = 0;
  size_t address_gap = 1; // the address distance between adjacent index
  for (int ix = index.size() - 1; ix >= 0; ix--) {
    address += index[ix] * address_gap;
    address_gap *= shape_[ix];
  }
  return tensor_->begin() + address;
}

template<typename dType>
const TensorIterator<dType> Tensor<dType>::get_const_iterator(const TensorIndex &index) const {
  size_t address = 0;
  size_t address_gap = 1; // the address distance between adjacent index
  for (int ix = index.size() - 1; ix >= 0; ix--) {
    address += index[ix] * address_gap;
    address_gap *= shape_[ix];
  }
  return tensor_->begin() + address;
}

template<typename dType>
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

template<typename dType>
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

template<typename dType>
void Tensor<dType>::set_value(const TensorIndex &index, const dType &value) {
  if (!is_index_valid(index)) {
    throw adg_exception::InvalidTensorIndexException(
      "InvalidTensorIndexException: the target shape is " + utils::vector_to_str(shape_) + " ,while getting index: "
        + utils::vector_to_str(index));
  }

  TensorIterator<dType> iter = get_iterator(index);
  *iter = value;
}

// when tensor has only one element
template<typename dType>
dType Tensor<dType>::get_value() const {
  if (size_ != 1) {
    throw adg_exception::InvalidTensorIndexException(
      "InvalidTensorIndexException: get_value() expects a tensor with single entry...");
  }

  return *tensor_->begin();
}

template<typename dType>
dType Tensor<dType>::get_value(const TensorIndex &index) const {
  if (!is_index_valid(index)) {
    throw adg_exception::InvalidTensorIndexException(
      "InvalidTensorIndexException: the target shape is " + utils::vector_to_str(shape_) + " ,while getting index: "
        + utils::vector_to_str(index));
  }

  return *get_const_iterator(index);
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
template<typename dType>
Tensor<dType> Tensor<dType>::take(const size_t &axis,
                                  const std::vector<size_t> &slice_indices) const {
  if (axis >= dim_) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> take: axis out of range");
  }

  for (size_t i : slice_indices) {
    if (i >= shape_[axis]) {
      throw adg_exception::IndexOutOfRangeError();
    }
  }

  size_t slice_len = slice_indices.size();
  TensorShape result_shape = shape_;
  result_shape[axis] = slice_len;
  Tensor<dType> result(std::move(result_shape));

  std::vector<size_t> indices = slice_indices;
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

template<typename dType>
Tensor<dType> Tensor<dType>::slice(const TensorSlice &slice) const {
  // slice is a vector of tuple3 {axis, start_index, end_index};
  // only support continuous slice
  TensorShape result_shape = shape_;
  TensorSlice sorted_slice(slice);
  std::sort(sorted_slice.begin(), sorted_slice.end(),
            [](const std::array<size_t, 3> &a, const std::array<size_t, 3> &b) { return a[0] < b[0]; });

  for (auto slice_tuple : sorted_slice) {
    if (slice_tuple[0] > dim_) {
      throw adg_exception::InvalidTensorSliceException(
        "Tensor >> slice: slice axis out of range: " + std::to_string(slice_tuple[0]));
    }

    if (slice_tuple[2] < slice_tuple[1]) {
      throw adg_exception::InvalidTensorSliceException(
        "Tensor >> slice: slice tuple (axis, start_ix, end_ix) must assure start_ix < end_ix");
    }

    result_shape[slice_tuple[0]] = slice_tuple[2] - slice_tuple[1];
  }

  Tensor<dType> result(std::move(result_shape));

  size_t dest_index = 0;
  slice_recursive_copy(0, 0, sorted_slice, get_tensor_const_ptr(),
                       result.get_tensor_ptr(), dest_index);
  return result;
}

template<typename dType>
void Tensor<dType>::slice_recursive_copy(const size_t &depth,
                                         const size_t &cur_axis,
                                         const TensorSlice &slice,
                                         const dType *src_ptr,
                                         dType *dest_ptr,
                                         size_t &dest_index) const {
  if (depth == slice.size()) {
    // reach the end of recursion, do the copy without restriction
    memcpy(dest_ptr + dest_index, src_ptr, sizeof(dType) * strides_[cur_axis - 1]);
    dest_index += strides_[cur_axis - 1];
    return;
  }

  // loop over the desired index
  size_t start_index, end_index;
  size_t depth_move = 0;
  if (cur_axis != slice[depth][0]) {
    start_index = 0;
    end_index = shape_[cur_axis];
  } else {
    // if cur axis equals the slice[depth][0], the depth should move by one slot
    depth_move = 1;
    start_index = slice[depth][1];
    end_index = slice[depth][2];
  }
  if (cur_axis == dim_ - 1) {
    // optimize the slice on the last dim
    // this part can get removed while the function will run expectedly
    size_t copy_len = end_index - start_index;
    memcpy(dest_ptr + dest_index, src_ptr + start_index, sizeof(dType) * copy_len);
    dest_index += copy_len;
    return;
  }
  for (size_t ix = start_index; ix < end_index; ++ix) {
    slice_recursive_copy(depth + depth_move,
                         cur_axis + 1,
                         slice,
                         src_ptr + ix * strides_[cur_axis],
                         dest_ptr,
                         dest_index);
  }
}

// copy returns a deep copy of current tensor
template<typename dType>
Tensor<dType> Tensor<dType>::copy() const {
  return Tensor<dType>(shape_, get_tensor_const_ptr());
}

template<typename dType>
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
      throw adg_exception::InvalidTensorShapeException(
        "Tensor >> do_shape_update: data size mismatch when update shape, "
        "new size is " +
          std::to_string(tmp_size) + ", while trying to keep the size of " +
          std::to_string(keep_size));
    }
  }

  size_ = tmp_size;
  strides_ = tmp_strides;
  shape_ = shape;
  dim_ = shape.size();
}

// reshape changes the shape_, dim_, strides_ of the tensor
template<typename dType>
void Tensor<dType>::reshape(const TensorShape &new_shape) {
  if (new_shape == shape_) {
    return;
  }

  if (!is_shape_valid(new_shape)) {
    throw adg_exception::InvalidTensorShapeException("Tensor ==> reshape");
  }

  // keep same size
  do_shape_update(new_shape, size_);
}

// helper function for transpose, computes the coordinate of an element on a
// specific axis given its vector index
template<typename dType>
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
template<typename dType>
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

template<typename dType>
size_t Tensor<dType>::get_index_after_concat(
  const size_t &ind, const size_t &axis, const size_t &offset_at_axis,
  const TensorShape &ori_strides, const TensorShape &new_strides) {
  // the offset between new index and ori index is definitely non-negative
  size_t offset = 0;
  {
    size_t ori_coord = get_coordinate_at_axis(ind, axis, ori_strides);
    size_t new_coord = ori_coord + offset_at_axis;

    offset += new_coord * new_strides[axis] - ori_coord * ori_strides[axis];
  }

  {
    size_t cur_coord;
    if (axis > 0) {
      for (int ax = axis - 1; ax >= 0; --ax) {
        cur_coord = get_coordinate_at_axis(ind, ax, ori_strides);
        offset += cur_coord * (new_strides[ax] - ori_strides[ax]);
      }
    }
  }

  size_t new_index = ind + offset;
  return static_cast<size_t>(new_index);
}

// do_transpose impl of transpose
// maps the index of original tensor to the index of the new tensor
template<typename dType>
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
template<typename dType>
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
template<typename dType>
Tensor<dType> Tensor<dType>::transpose() const {
  if (dim_ <= 1) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> transpose: not enough dimension");
  }

  return transpose(dim_ - 2, dim_ - 1);
}

// short for transpose
template<typename dType>
Tensor<dType> Tensor<dType>::t() const {
  if (dim_ <= 1) {
    throw adg_exception::AxisOutOfRangeError(
      "Tensor >> t: not enough dimension");
  }

  return transpose(dim_ - 2, dim_ - 1);
}

template<typename dType>
void Tensor<dType>::fill_diag(const std::vector<dType> &diag_values) {
  if (dim_ != 2) {
    throw adg_exception::InvalidTensorShapeException(
      "Tensor >> fill_diag get non-2d tensor");
  }
  size_t diag_len = diag_values.size();
  size_t tensor_min_dim_len = std::min(shape_[0], shape_[1]);

  if (diag_len > tensor_min_dim_len) {
    throw adg_exception::MismatchTensorShapeError(
      "MismatchTensorShapeError >> fill_diag of len " +
        std::to_string(diag_len) + " , longer than shape_min " +
        std::to_string(tensor_min_dim_len));
  }

  utils::math::fill_diagonal(std::min(shape_[0], diag_len), shape_[1],
                             &*diag_values.begin(), get_tensor_ptr());
}

template<typename dType>
Tensor<int32_t> Tensor<dType>::to_int() const {
  std::vector<int32_t> values(tensor_->begin(), tensor_->end());
  return Tensor<int32_t>(shape_, std::move(values));
}

template<typename dType>
Tensor<float> Tensor<dType>::to_float() const {
  std::vector<float> values(tensor_->begin(), tensor_->end());
  return Tensor<float>(shape_, std::move(values));
}

template<typename dType>
Tensor<double> Tensor<dType>::to_double() const {
  std::vector<double> values(tensor_->begin(), tensor_->end());
  return Tensor<double>(shape_, std::move(values));
}

template<typename dType>
void Tensor<dType>::map(Mapper<dType> &mapper) {
#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  mapper.run(get_tensor_ptr(), size_);
#endif
}

template<typename dType>
void Tensor<dType>::map(Mapper<dType> &&mapper) {
#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  mapper.run(get_tensor_ptr(), size_);
#endif
}

template<typename dType>
void Tensor<dType>::map(const std::function<void(dType &)> &func) {
  Mapper<dType> mapper(func);

#if ADGC_MULTI_THREADS_NUM_
  utils::threads::ThreadPool pool;
  pool.start(ADGC_MULTI_THREADS_NUM_);
  mapper.run(get_tensor_ptr(), size_, &pool);
#else
  mapper.run(get_tensor_ptr(), size_);
#endif
}

}