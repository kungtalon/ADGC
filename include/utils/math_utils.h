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

void gemm(const size_t M, const size_t N, const size_t K, const double *mat_a,
          const double *mat_b, double *mat_c);

void gemm(const size_t M, const size_t N, const size_t K, const float *mat_a,
          const float *mat_b, float *mat_c);

void gemm(const size_t M, const size_t N, const size_t K, const int32_t *mat_a,
          const int32_t *mat_b, int32_t *mat_c);

template <typename dType>
void elementwise_multiply(const size_t &size, const dType *mat_a,
                          const dType *mat_b, dType *mat_c);

template <typename dType>
void elementwise_add(const size_t &size, const dType *mat_a, const dType *mat_b,
                     dType *mat_c, bool subtract = false);

} // namespace math
} // namespace utils

#include "utils/math_utils.tcc"

#endif