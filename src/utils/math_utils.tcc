#ifndef ADGC_UTILS_MATH_UTILS_TCC_
#define ADGC_UTILS_MATH_UTILS_TCC_

#include "utils/math_utils.h"

namespace utils {
namespace math {

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