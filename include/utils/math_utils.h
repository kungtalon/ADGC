#ifndef ADGC_UTILS_MATH_UTILS_H_
#define ADGC_UTILS_MATH_UTILS_H_

#include <cblas.h>
#include <iostream>
#include <stdlib.h>

#include "exception/exception.h"
#include "thread.h"
#include "utils.h"

namespace utils {
namespace math {

template <typename dType>
void tensor_gemm(const size_t &size_a, const size_t &size_b,
                 const size_t &size_c, const size_t &M, const size_t &N,
                 const size_t &K, const dType *mat_a, const dType *mat_b,
                 dType *mat_c);

inline void gemm(const size_t M, const size_t N, const size_t K,
                 const float *mat_a, const float *mat_b, float *mat_c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

inline void gemm(const size_t M, const size_t N, const size_t K,
                 const double *mat_a, const double *mat_b, double *mat_c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

inline void gemm(const size_t M, const size_t N, const size_t K,
                 const int32_t *mat_a, const int32_t *mat_b, int32_t *mat_c) {
  throw adg_exception::NonImplementedException();
}

// elementwise mult
inline void elementwise_multiply(const size_t &size, const double *mat_a,
                                 const double *mat_b, double *mat_c) {
  // memcpy(mat_c, mat_b, sizeof(double) * size);
  // cblas_ddot(size, mat_a, 1, mat_c, 1);
  for (int ix = 0; ix < size; ++ix) {
    mat_c[ix] = mat_a[ix] * mat_b[ix];
  }
}

inline void elementwise_multiply(const size_t &size, const float *mat_a,
                                 const float *mat_b, float *mat_c) {
  // memcpy(mat_c, mat_b, sizeof(float) * size);
  // cblas_sdot(size, mat_a, 1, mat_c, 1);
  for (int ix = 0; ix < size; ++ix) {
    mat_c[ix] = mat_a[ix] * mat_b[ix];
  }
}

inline void elementwise_multiply(const size_t &size, const int32_t *mat_a,
                                 const int32_t *mat_b, int32_t *mat_c) {
  throw adg_exception::NonImplementedException();
}

// elementwise addition
inline void elementwise_add(const size_t &size, const double *mat_a,
                            const double *mat_b, double *mat_c,
                            bool subtract = false) {
  if (subtract) {
    memcpy(mat_c, mat_a, sizeof(double) * size);
    cblas_daxpy(size, -1., mat_b, 1, mat_c, 1);
  } else {
    memcpy(mat_c, mat_b, sizeof(double) * size);
    cblas_daxpy(size, 1., mat_a, 1, mat_c, 1);
  }
}

inline void elementwise_add(const size_t &size, const float *mat_a,
                            const float *mat_b, float *mat_c,
                            bool subtract = false) {
  if (subtract) {
    memcpy(mat_c, mat_a, sizeof(float) * size);
    cblas_saxpy(size, -1., mat_b, 1, mat_c, 1);
  } else {
    memcpy(mat_c, mat_b, sizeof(float) * size);
    cblas_saxpy(size, 1., mat_a, 1, mat_c, 1);
  }
}

inline void elementwise_add(const size_t &size, const int32_t *mat_a,
                            const int32_t *mat_b, int32_t *mat_c,
                            bool subtract = false) {
  throw adg_exception::NonImplementedException();
}

inline void fill_diagonal(const size_t &M, const size_t &N,
                          const double *values, double *mat) {
  cblas_daxpy(std::min(M, N), 1., values, 1, mat, N + 1);
}

inline void fill_diagonal(const size_t &M, const size_t &N, const float *values,
                          float *mat) {
  cblas_saxpy(std::min(M, N), 1., values, 1, mat, N + 1);
}

inline void fill_diagonal(const size_t &M, const int32_t &N,
                          const int32_t *values, int32_t *mat) {
  throw adg_exception::NonImplementedException();
}
} // namespace math
} // namespace utils

#include "utils/math_utils.tcc"

#endif