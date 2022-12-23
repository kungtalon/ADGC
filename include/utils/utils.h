#ifndef ADGC_UTILS_UTILS_H_
#define ADGC_UTILS_UTILS_H_

#include <sstream>
#include <unordered_map>
#include <vector>
#include <regex>
#include <string>
#include <fstream>

namespace utils {

// convert a matrix to string
template<typename T>
std::string array_to_str(size_t m, size_t n, T *p, std::string name = "");

// print the tensor recursively from the innermost axis to the outermost
template<typename T>
std::string multi_array_to_str(const std::vector<size_t> &shape, T *arr_p,
                               const std::string &name = "");

template<typename T>
void multi_array_to_str_helper(size_t rank, size_t cur_axis,
                               const std::vector<size_t> &shape,
                               const std::vector<size_t> &axis_inc, T *p_start,
                               T *p_end, std::stringstream &out);

class TypeCounter {
 public:
  inline size_t inc(const std::string &type) {
    if (counter_.find(type) == counter_.end()) {
      counter_[type] = 1;
      return 0;
    } else {
      return counter_[type]++;
    }
  }

  inline void clear() { counter_.clear(); }

 private:
  std::unordered_map<std::string, size_t> counter_;
};

void read_lines_from_file(const char *file_name, std::vector<std::string> &output_vec);

std::vector<std::string> str_split(const std::string &str, const std::string &sep);

std::vector<std::string> str_split(const std::string &str, const std::regex &re);

} // namespace utils

#include "utils/utils.tcc"

#endif
