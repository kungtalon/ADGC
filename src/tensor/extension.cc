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

  size_t src_index = 0, dest_index = 0;

  for (int ix = 0; ix < src_tensor.get_size() / src_ncol; ++ix) {
    cblas_dcopy(src_ncol, src_ptr + src_index, 1, dest_ptr + dest_index, gaps[1] + 1);
    src_index += src_ncol;
    if ((ix + 1) % src_nrow != 0) {
      dest_index += result_ncol * (gaps[0] + 1);
    } else {
      dest_index += result_ncol;
    }
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

template<>
double squared_sum(const Tensor<double> &ts) {
  return cblas_ddot(ts.get_size(), ts.get_tensor_const_ptr(), 1, ts.get_tensor_const_ptr(), 1);
}

template<>
float squared_sum(const Tensor<float> &ts) {
  return cblas_sdot(ts.get_size(), ts.get_tensor_const_ptr(), 1, ts.get_tensor_const_ptr(), 1);
}

template<>
Tensor<float> sqrt(const Tensor<float> &ts) {
  Tensor<float> result(ts.get_shape());
  vsSqrt(ts.get_size(), ts.get_tensor_const_ptr(), &*result.get_iterator());
  return result;
}

template<>
Tensor<double> sqrt(const Tensor<double> &ts) {
  Tensor<double> result(ts.get_shape());
  vdSqrt(ts.get_size(), ts.get_tensor_const_ptr(), &*result.get_iterator());
  return result;
}

template<>
Tensor<float> add_vec(const Tensor<float> &lt,
                      const Tensor<float> &rt,
                      const size_t &axis) {
  Tensor<float> vec, mat;
  if (lt.get_dim() == 1) {
    vec = lt;
    mat = rt;
  } else if (rt.get_dim() == 1) {
    vec = rt;
    mat = lt;
  } else {
    throw adg_exception::InvalidTensorShapeException("Tensor >> add_vec: expect one operand with dim 1...");
  }

  if (mat.get_shape(axis) != vec.get_size()) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> add_vec: expect two operands to have matched size in the axis...");
  }

  Tensor<float> result = mat.copy();

  const float *vector_ptr = vec.get_tensor_const_ptr();
  size_t mat_stride = result.get_stride(axis);
  size_t vec_size = vec.get_size();

  size_t matrix_index = 0, vector_index = 0;
  while (matrix_index < mat.get_size()) {
    cblas_saxpy(mat_stride,
                1.0,
                vector_ptr + vector_index,
                0,
                &*(result.get_iterator() + matrix_index),
                1);
    vector_index = (vector_index + 1) % vec_size;
    matrix_index += mat_stride;
  }
  return result;
}

template<>
Tensor<double> add_vec(const Tensor<double> &lt,
                       const Tensor<double> &rt,
                       const size_t &axis) {
  Tensor<double> vec, mat;
  if (lt.get_dim() == 1) {
    vec = lt;
    mat = rt;
  } else if (rt.get_dim() == 1) {
    vec = rt;
    mat = lt;
  } else {
    throw adg_exception::InvalidTensorShapeException("Tensor >> add_vec: expect one operand with dim 1...");
  }

  if (mat.get_shape(axis) != vec.get_size()) {
    throw adg_exception::MismatchTensorShapeError(
      "Tensor >> add_vec: expect two operands to have matched size in the axis...");
  }

  Tensor<double> result = mat.copy();

  const double *vector_ptr = vec.get_tensor_const_ptr();
  size_t mat_stride = result.get_stride(axis);
  size_t vec_size = vec.get_size();

  size_t matrix_index = 0, vector_index = 0;
  while (matrix_index < mat.get_size()) {
    cblas_daxpy(mat_stride,
                1.0,
                vector_ptr + vector_index,
                0,
                &*(result.get_iterator() + matrix_index),
                1);
    vector_index = (vector_index + 1) % vec_size;
    matrix_index += mat_stride;
  }
  return result;
}

}