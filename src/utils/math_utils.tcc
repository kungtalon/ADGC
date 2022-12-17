#include "utils/math_utils.h"

namespace utils {
namespace math {

void gemm(const size_t M, const size_t N, const size_t K, const float *mat_a,
          const float *mat_b, float *mat_c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

void gemm(const size_t M, const size_t N, const size_t K, const double *mat_a,
          const double *mat_b, double *mat_c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., mat_a, K,
              mat_b, N, 0., mat_c, N);
}

void gemm(const size_t M, const size_t N, const size_t K, const int32_t *mat_a,
          const int32_t *mat_b, int32_t *mat_c) {
  throw adg_exception::NonImplementedException();
}

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
void elementwise_multiply(const size_t &size, const dType *mat_a,
                          const dType *mat_b, dType *mat_c) {
  // TODO: replace with MLK vsMul
  // big precision problem!
  for (size_t ix = 0; ix < size; ix++) {
    mat_c[ix] = mat_a[ix] * mat_b[ix];
  }
}

template <typename dType>
void elementwise_add(const size_t &size, const dType *mat_a, const dType *mat_b,
                     dType *mat_c, bool subtract) {
  // TODO: replace with ..?
  if (subtract) {
    for (size_t ix = 0; ix < size; ix++) {
      mat_c[ix] = mat_a[ix] - mat_b[ix];
    }
  } else {
    for (size_t ix = 0; ix < size; ix++) {
      mat_c[ix] = mat_a[ix] + mat_b[ix];
    }
  }
}

// explicit instantiation of template function;
template void tensor_gemm<double>(const size_t &size_a, const size_t &size_b,
                                  const size_t &size_c, const size_t &M,
                                  const size_t &N, const size_t &K,
                                  const double *mat_a, const double *mat_b,
                                  double *mat_c);

template void tensor_gemm<float>(const size_t &size_a, const size_t &size_b,
                                 const size_t &size_c, const size_t &M,
                                 const size_t &N, const size_t &K,
                                 const float *mat_a, const float *mat_b,
                                 float *mat_c);

template void tensor_gemm<int32_t>(const size_t &size_a, const size_t &size_b,
                                   const size_t &size_c, const size_t &M,
                                   const size_t &N, const size_t &K,
                                   const int32_t *mat_a, const int32_t *mat_b,
                                   int32_t *mat_c);

template void elementwise_multiply(const size_t &size, const double *mat_a,
                                   const double *mat_b, double *mat_c);
template void elementwise_multiply(const size_t &size, const float *mat_a,
                                   const float *mat_b, float *mat_c);
template void elementwise_multiply(const size_t &size, const int32_t *mat_a,
                                   const int32_t *mat_b, int32_t *mat_c);

template void elementwise_add(const size_t &size, const double *mat_a,
                              const double *mat_b, double *mat_c,
                              bool subtract);
template void elementwise_add(const size_t &size, const float *mat_a,
                              const float *mat_b, float *mat_c, bool subtract);
template void elementwise_add(const size_t &size, const int32_t *mat_a,
                              const int32_t *mat_b, int32_t *mat_c,
                              bool subtract);

} // namespace math
} // namespace utils
