#ifndef ADGC_UTILS_MATH_UTILS_H_
#define ADGC_UTILS_MATH_UTILS_H_

#include <cblas.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

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

template <typename dType>
void tensor_kron_product(const size_t &size_a, const size_t &size_b,
                         const size_t &row_a, const size_t &col_a,
                         const size_t &row_b, const size_t &col_b,
                         const dType *mat_a, const dType *mat_b, dType *mat_c);
} // namespace math
} // namespace utils

#include "utils/math_utils.tcc"

#endif