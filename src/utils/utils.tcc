#ifndef ADGC_UTILS_UTILS_TCC_
#define ADGC_UTILS_UTILS_TCC_

#include "utils/utils.h"

namespace utils {

template <typename T, typename A>
std::string vector_to_str(const std::vector<T, A> &vec) {
  std::stringstream ss;
  for (auto val : vec) {
    ss << std::to_string(val) << " ";
  }
  return ss.str();
}

// convert a matrix to string
template <typename T>
std::string array_to_str(size_t m, size_t n, T *p, std::string name) {
  std::stringstream ss;
  if (!name.empty()) {
    ss << "Matrix : " << name << std::endl;
  }
  for (int i = 0; i < m; i++) {
    ss << "[ ";
    for (int j = 0; j < n; j++) {
      ss << p[i * n + j];
      if (j < n - 1) {
        ss << "\t";
      }
    }
    ss << " ]";
    if (i < m - 1) {
      ss << std::endl;
    }
  }
  return ss.str();
}

template <typename T>
std::string multi_array_to_str(const std::vector<size_t> &shape, T *arr_p,
                               const std::string &name) {
  std::stringstream ss;
  if (!name.empty()) {
    ss << "Matrix : " << name << std::endl;
  }

  size_t rank = shape.size();

  if (rank == 0) {
    return "";
  } else if (rank == 1) {
    return array_to_str(1, shape[0], arr_p, name);
  } else if (rank == 2) {
    return "[ " + array_to_str(shape[0], shape[1], arr_p, name) + " ]";
  }

  {
    auto strides = std::vector<size_t>(rank, 1);
    for (int ix = rank - 2; ix >= 0; ix--) {
      strides[ix] = strides[ix + 1] * shape[ix + 1];
    }

    size_t size = strides[0] * shape[0];

    multi_array_to_str_helper(rank, 0, shape, strides, arr_p, arr_p + size, ss);
  }

  return ss.str();
}

template <typename T>
void multi_array_to_str_helper(size_t rank, size_t cur_axis,
                               const std::vector<size_t> &shape,
                               const std::vector<size_t> &strides, T *p_start,
                               T *p_end, std::stringstream &out) {
  if (cur_axis == rank - 2) {
    out << "[ " << array_to_str(shape[rank - 2], shape[rank - 1], p_start)
        << " ]";
    return;
  }

  out << "[ ";
  T *cur_p_start = p_start;
  T *cur_p_end = p_start + strides[cur_axis];
  while (cur_p_end <= p_end) {
    // out << "Loop " << cur_axis << " " << std::endl;
    multi_array_to_str_helper(rank, cur_axis + 1, shape, strides, cur_p_start,
                              cur_p_end, out);
    cur_p_start = cur_p_end;
    cur_p_end += strides[cur_axis];
    if (cur_p_end <= p_end) {
      out << std::endl;
    }
  }
  out << " ]";
}


// template instantiations
template std::string array_to_str<int32_t>(size_t m, size_t n, int32_t *p,
                                           std::string name = "");
template std::string array_to_str<float>(size_t m, size_t n, float *p,
                                         std::string name = "");
template std::string array_to_str<double>(size_t m, size_t n, double *p,
                                          std::string name = "");
// template std::string array_to_str<bool>(size_t m, size_t n, bool *p,
//                                         std::string name = "");
template std::string
multi_array_to_str<int32_t>(const std::vector<size_t> &shape, int32_t *arr_p,
                            const std::string &name = "");

template std::string multi_array_to_str<float>(const std::vector<size_t> &shape,
                                               float *arr_p,
                                               const std::string &name = "");

template void multi_array_to_str_helper<int32_t>(
    size_t rank, size_t cur_axis, const std::vector<size_t> &shape,
    const std::vector<size_t> &axis_inc, int32_t *p_start, int32_t *p_end,
    std::stringstream &out);

template void multi_array_to_str_helper<float>(
    size_t rank, size_t cur_axis, const std::vector<size_t> &shape,
    const std::vector<size_t> &axis_inc, float *p_start, float *p_end,
    std::stringstream &out);

template void multi_array_to_str_helper<double>(
    size_t rank, size_t cur_axis, const std::vector<size_t> &shape,
    const std::vector<size_t> &axis_inc, double *p_start, double *p_end,
    std::stringstream &out);

// template void multi_array_to_str_helper<bool>(
//     size_t rank, size_t cur_axis, const std::vector<size_t> &shape,
//     const std::vector<size_t> &axis_inc, bool *p, std::stringstream &out);

} // namespace utils

#endif