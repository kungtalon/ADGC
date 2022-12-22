#ifndef ADGC_UTILS_MATH_UTILS_TCC_
#define ADGC_UTILS_MATH_UTILS_TCC_

#include "utils/math_utils.h"

namespace utils {
namespace math {

inline void gemm(const size_t &M, const size_t &N, const size_t &K,
                 const float *mat_a, const float *mat_b, float *mat_c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

inline void gemm(const size_t &M, const size_t &N, const size_t &K,
                 const double *mat_a, const double *mat_b, double *mat_c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

inline void gemm(const size_t &M, const size_t &N, const size_t &K,
                 const int32_t *mat_a, const int32_t *mat_b, int32_t *mat_c) {
  throw adg_exception::NonImplementedException();
}

inline void kron1d(const size_t &size_a, const size_t &size_b,
                   const size_t &size_c, const double *mat_a,
                   const double *mat_b, double *mat_c) {
  cblas_dger(CblasRowMajor, size_a, size_b, 1.0, mat_a, 1, mat_b, 1, mat_c,
             size_b);
}

inline void kron1d(const size_t &size_a, const size_t &size_b,
                   const size_t &size_c, const float *mat_a, const float *mat_b,
                   float *mat_c) {
  cblas_sger(CblasRowMajor, size_a, size_b, 1.0, mat_a, 1, mat_b, 1, mat_c,
             size_b);
}

inline void kron1d(const size_t &size_a, const size_t &size_b,
                   const size_t &size_c, const int32_t *mat_a,
                   const int32_t *mat_b, int32_t *mat_c) {
  throw adg_exception::NonImplementedException();
}

inline void kron2d(const size_t &row_a, const size_t &col_a,
                   const size_t &row_b, const size_t &col_b,
                   const double *mat_a, const double *mat_b, double *mat_c) {
  size_t index_a = 0;
  size_t index_b = 0;
  size_t index_c = 0;
  size_t col_c = col_a * col_b;
  for (size_t ix = 0; ix < row_a; ++ix) {
    index_b = 0;
    for (size_t j = 0; j < row_b; ++j) {
      kron1d(col_a, col_b, col_c, mat_a + index_a, mat_b + index_b,
             mat_c + index_c);
      index_b += col_b;
      index_c += col_c;
    }
    index_a += col_a;
  }
}

inline void kron2d(const size_t &row_a, const size_t &col_a,
                   const size_t &row_b, const size_t &col_b, const float *mat_a,
                   const float *mat_b, float *mat_c) {
  size_t index_a = 0;
  size_t index_b = 0;
  size_t index_c = 0;
  size_t col_c = col_a * col_b;
  for (size_t ix = 0; ix < row_a; ++ix) {
    index_b = 0;
    for (size_t j = 0; j < row_b; ++j) {
      kron1d(col_a, col_b, col_c, mat_a + index_a, mat_b + index_b,
             mat_c + index_c);
      index_b += col_b;
      index_c += col_c;
    }
    index_a += col_a;
  }
}

inline void kron2d(const size_t &size_a, const size_t &size_b,
                   const size_t &size_c, const int32_t *mat_a,
                   const int32_t *mat_b, int32_t *mat_c) {
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
    cblas_dcopy(size, mat_a, 1, mat_c, 1);
    cblas_daxpy(size, -1., mat_b, 1, mat_c, 1);
  } else {
    cblas_dcopy(size, mat_b, 1, mat_c, 1);
    cblas_daxpy(size, 1., mat_a, 1, mat_c, 1);
  }
}

inline void elementwise_add(const size_t &size, const float *mat_a,
                            const float *mat_b, float *mat_c,
                            bool subtract = false) {
  if (subtract) {
    cblas_scopy(size, mat_a, 1, mat_c, 1);
    cblas_saxpy(size, -1., mat_b, 1, mat_c, 1);
  } else {
    cblas_scopy(size, mat_b, 1, mat_c, 1);
    cblas_saxpy(size, 1., mat_a, 1, mat_c, 1);
  }
}

inline void elementwise_add(const size_t &size, const int32_t *mat_a,
                            const int32_t *mat_b, int32_t *mat_c,
                            bool subtract = false) {
  throw adg_exception::NonImplementedException();
}

inline void elementwise_add_inplace(const size_t &size, double *mat_a,
                                    const double *mat_b,
                                    bool subtract = false) {
  // result should be stored in mat_a
  if (subtract) {
    cblas_daxpy(size, -1., mat_b, 1, mat_a, 1);
  } else {
    cblas_daxpy(size, 1., mat_b, 1, mat_a, 1);
  }
}

inline void elementwise_add_inplace(const size_t &size, float *mat_a,
                                    const float *mat_b, bool subtract = false) {
  if (subtract) {
    cblas_saxpy(size, -1., mat_b, 1, mat_a, 1);
  } else {
    cblas_saxpy(size, 1., mat_b, 1, mat_a, 1);
  }
}

inline void elementwise_add_inplace(const size_t &size, int32_t *mat_a,
                                    const int32_t *mat_b,
                                    bool subtract = false) {
  throw adg_exception::NonImplementedException();
}

inline void elementwise_addn(const size_t &size, double *mat,
                             const double &number, bool subtract = false) {
  double *tmp = new double[2];
  tmp[0] = number;
  if (subtract) {
    cblas_daxpy(size, -1., tmp, 0, mat, 1);
  } else {
    cblas_daxpy(size, 1., tmp, 0, mat, 1);
  }
  delete tmp;
}

inline void elementwise_addn(const size_t &size, float *mat,
                             const float &number, bool subtract = false) {
  float *tmp = new float[2];
  tmp[0] = number;
  if (subtract) {
    cblas_saxpy(size, -1., tmp, 0, mat, 1);
  } else {
    cblas_saxpy(size, 1., tmp, 0, mat, 1);
  }
  delete tmp;
}

inline void elementwise_addn(const size_t &size, int32_t *mat,
                             const int32_t &number, bool subtract = false) {
  throw adg_exception::NonImplementedException();
}

inline void elementwise_negative(const size_t &size, double *mat) {
  double *tmp = new double[2];
  tmp[0] = 0.;
  cblas_daxpby(size, 1., tmp, 0, -1., mat, 1);
  delete tmp;
}

inline void elementwise_negative(const size_t &size, float *mat) {
  float *tmp = new float[2];
  tmp[0] = 0.;
  cblas_saxpby(size, 1., tmp, 0, -1., mat, 1);
  delete tmp;
}

inline void elementwise_negative(const size_t &size, const int32_t *mat) {
  throw ::adg_exception::NonImplementedException();
}

inline float sum(const size_t &size, const float *mat_a, const size_t &inc) {
  float constant = 1;
  return cblas_sdot(size, mat_a, inc, &constant, 0);
}

inline double sum(const size_t &size, const double *mat_a, const size_t &inc) {
  double constant = 1;
  return cblas_ddot(size, mat_a, inc, &constant, 0);
}

inline int32_t sum(const size_t &size, const int32_t *mat_a,
                   const size_t &inc) {
  throw adg_exception::NonImplementedException();
}

inline void fill_diagonal(const size_t &M, const size_t &N,
                          const double *values, double *mat) {
  cblas_dcopy(std::min(M, N), values, 1, mat, N + 1);
}

inline void fill_diagonal(const size_t &M, const size_t &N, const float *values,
                          float *mat) {
  cblas_scopy(std::min(M, N), values, 1, mat, N + 1);
}

inline void fill_diagonal(const size_t &M, const int32_t &N,
                          const int32_t *values, int32_t *mat) {
  throw adg_exception::NonImplementedException();
}

// math functions for loss functions

inline double sigmoid(double x) {
  return 1 / (1 + std::exp(std::min(-x, 100.)));
}

inline float sigmoid(float x) {
  return 1 / (1 + std::exp(std::min(-x, (float)100)));
}

inline double relu(double x) { return std::max(x, 0.); }

inline float relu(float x) { return std::max(x, (float)0); }

template <typename dType>
void tensor_gemm(const size_t &size_a, const size_t &size_b,
                 const size_t &size_c, const size_t &M, const size_t &N,
                 const size_t &K, const dType *mat_a, const dType *mat_b,
                 dType *mat_c) {
  // shape_a : [..., M, K]
  // shape_b : [..., K, N]
  // shape_c : [..., M, N]
  size_t inc_a = M * K;
  size_t inc_b = K * N;
  size_t inc_c = M * N;
  size_t n_blocks = size_a / inc_a;

  if (n_blocks == 1) {
    gemm(M, N, K, mat_a, mat_b, mat_c);
  }
#if USE_MULTI_THREAD_GEMM_BOOL
  else {
    auto threads_p = new utils::threads::ThreadPool();
    threads_p->start(2);

    // split the tensor into blocks of size M*N or N*K
    // use thread pool to run the gemm concurrently
    dType *cur_c_p = mat_c;
    for (size_t ix = 0; ix < n_blocks; ix++) {
      const dType *cur_a_p = mat_a + inc_a * ix;
      const dType *cur_b_p = mat_b + inc_b * ix;
      threads_p->submit_job(
          [&]() mutable { gemm(M, N, K, cur_a_p, cur_b_p, cur_c_p); });
      cur_c_p += inc_c;
    }

    while (true) {
      if (!threads_p->busy()) {
        break;
      }
    }
  }
#else
  else {
    // split the tensor into blocks of size M*N or N*K
    dType *cur_c_p = mat_c;
    for (size_t ix = 0; ix < n_blocks; ix++) {
      const dType *cur_a_p = mat_a + inc_a * ix;
      const dType *cur_b_p = mat_b + inc_b * ix;
      gemm(M, N, K, cur_a_p, cur_b_p, cur_c_p);
      cur_c_p += inc_c;
    }
  }
#endif
}

template <typename dType>
void tensor_kron_product(const size_t &size_a, const size_t &size_b,
                         const size_t &row_a, const size_t &col_a,
                         const size_t &row_b, const size_t &col_b,
                         const dType *mat_a, const dType *mat_b, dType *mat_c) {
  // shape_a : [..., MA, NA]
  // shape_b : [..., MB, NB]
  // shape_c : [..., MA*MB, NA*NB]
  size_t inc_a = row_a * col_a;
  size_t inc_b = row_b * col_b;
  size_t inc_c = inc_a * inc_b;
  size_t n_blocks = size_a / inc_a;

  if (n_blocks == 1) {
    kron2d(row_a, col_a, row_b, col_b, mat_a, mat_b, mat_c);
  } else {
    // split the tensor into blocks of size M*N or N*K
    dType *cur_c_p = mat_c;
    size_t index_a = 0;
    size_t index_b = 0;
    for (size_t ix = 0; ix < n_blocks; ix++) {
      const dType *cur_a_p = mat_a + inc_a;
      const dType *cur_b_p = mat_b + inc_b;
      kron2d(row_a, col_a, row_b, col_b, cur_a_p, cur_b_p, cur_c_p);
      index_a += inc_a;
      index_b += inc_b;
      cur_c_p += inc_c;
    }
  }
}

} // namespace math
} // namespace utils

#endif