//
// Created by kungtalon on 2022/12/25.
//

#include "tensor/extension.h"

namespace tensor {

template<>
Tensor<double> dilate2d(const Tensor<double> &src_tensor,
                        const std::array<size_t, 2> &gaps,
                        const double &value) {
  // gaps are the numbers of values that we want to fill in between two adjacent elements
  // if the gap for column-wise is 2, the original size is n, then the result size is (n-1)gap + n
  if (!gaps[0] && !gaps[1]) {
    return src_tensor;
  }

  size_t dim = src_tensor.get_dim();
  tensor::TensorShape src_shape = src_tensor.get_shape();
  size_t src_nrow = src_shape[dim - 2];
  size_t src_ncol = src_shape[dim - 1];
  size_t result_nrow = (src_nrow - 1) * gaps[0] + src_shape[dim - 2];
  size_t result_ncol = (src_ncol - 1) * gaps[1] + src_shape[dim - 1];

  TensorShape result_shape(std::move(src_shape));
  result_shape[dim - 2] = result_nrow;
  result_shape[dim - 1] = result_ncol;

  Tensor<double> result(std::move(result_shape), value);

  double *dest_ptr = &*result.get_iterator();
  const double *src_ptr = src_tensor.get_tensor_const_ptr();

  if (gaps[0] == 0) {
    cblas_dcopy(src_tensor.get_size(), src_ptr, 1, dest_ptr, gaps[1] + 1);
    return result;
  }

  size_t src_index = 0;
  for (int ix = 0; ix < src_nrow; ++ix) {
    cblas_dcopy(src_tensor.get_size(), src_ptr + src_index, 1, dest_ptr, gaps[1] + 1);
    src_index += src_ncol;
    dest_ptr += result_ncol * (gaps[0] + 1);
  }
  return result;
}

template<>
Tensor<float> dilate2d(const Tensor<float> &src_tensor,
                       const std::array<size_t, 2> &gaps,
                       const float &value) {
  // gaps are the numbers of values that we want to fill in between two adjacent elements
  // if the gap for column-wise is 2, the original size is n, then the result size is (n-1)gap + n
  if (!gaps[0] && !gaps[1]) {
    return src_tensor;
  }

  size_t dim = src_tensor.get_dim();
  tensor::TensorShape src_shape = src_tensor.get_shape();
  size_t src_nrow = src_shape[dim - 2];
  size_t src_ncol = src_shape[dim - 1];
  size_t result_nrow = (src_nrow - 1) * gaps[0] + src_shape[dim - 2];
  size_t result_ncol = (src_ncol - 1) * gaps[1] + src_shape[dim - 1];

  TensorShape result_shape(std::move(src_shape));
  result_shape[dim - 2] = result_nrow;
  result_shape[dim - 1] = result_ncol;

  Tensor<float> result(std::move(result_shape), value);

  float *dest_ptr = &*result.get_iterator();
  const float *src_ptr = src_tensor.get_tensor_const_ptr();

  size_t src_index = 0, dest_index = 0;

  for (int ix = 0; ix < src_tensor.get_size() / src_ncol; ++ix) {
    cblas_scopy(src_ncol, src_ptr + src_index, 1, dest_ptr + dest_index, gaps[1] + 1);
    src_index += src_ncol;
    if ((ix + 1) % src_nrow != 0) {
      dest_index += result_ncol * (gaps[0] + 1);
    } else {
      dest_index += result_ncol;
    }
  }

  return result;
}

}