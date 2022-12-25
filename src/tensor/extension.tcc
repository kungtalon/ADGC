#include "tensor/extension.h"

namespace tensor {

template<typename dType>
Tensor<dType> pad2d(const Tensor<dType> &src_tensor,
                    const std::vector<size_t> &paddings,
                    const dType &value) {
  return pad2d(src_tensor, {{paddings[0], paddings[0]}, {paddings[1], paddings[1]}}, value);
}

template<typename dType>
Tensor<dType> pad2d(const Tensor<dType> &src_tensor,
                    const std::vector<std::pair<size_t, size_t>> &paddings,
                    const dType &value) {
  size_t dim = src_tensor.get_dim();
  TensorShape result_shape = src_tensor.get_shape();
  result_shape[dim - 1] += paddings[1].first + paddings[1].second;
  result_shape[dim - 2] += paddings[0].first + paddings[0].second;

  size_t new_size = 1;
  std::vector<size_t> new_strides(dim);
  for (int ix = dim - 1; ix >= 0; --ix) {
    new_strides[ix] = new_size;
    new_size *= result_shape[ix];
  }

  auto result_values = std::vector<dType>(new_size, value);

  dType *dest_ptr = &*result_values.begin();
  const dType *src_ptr = src_tensor.get_tensor_const_ptr();

  size_t src_column_size = src_tensor.get_shape()[dim - 1];
  size_t dest_index = 0, src_index = 0, dest_offset, src_offset;
  size_t dest_inc = dim == 2 ? new_size : new_strides[dim - 3];
  size_t src_inc = dim == 2 ? src_tensor.get_size() : src_tensor.get_strides()[dim - 3];
  while (dest_index < new_size) {
    // loop for batch
    dest_offset = paddings[0].first * new_strides[dim - 2] + paddings[1].first;
    src_offset = 0;
    while (src_offset < src_inc) {
      // loop for copying each row
      memcpy(dest_ptr + dest_index + dest_offset,
             src_ptr + src_index + src_offset,
             sizeof(dType) * src_column_size);
      src_offset += src_column_size;
      dest_offset += new_strides[dim - 2];
    }
    dest_index += dest_inc;
    src_index += src_inc;
  }

  Tensor<dType> result(result_shape, result_values);
  return result;
}

}