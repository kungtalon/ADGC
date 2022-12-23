//
// Created by Zelon on 2022/12/22.
//

#include "utils/utils.h"

namespace utils {

void read_lines_from_file(const char *file_name, std::vector<std::string> &output_vec) {
  std::ifstream infile(file_name);
  std::string line;

  std::regex blank_pattern("\\s+");
  const char *extra_space = " \t\n\r\f\v";

  while (std::getline(infile, line)) {
    if (line.empty() || std::regex_match(line.c_str(), blank_pattern)) {
      continue;
    }
    line.erase(0, line.find_first_not_of(extra_space));
    line.erase(line.find_last_not_of(extra_space) + 1);
    output_vec.push_back(line);
  }
}

std::vector<std::string> str_split(const std::string &str, const std::string &sep) {
  std::regex regex{sep};
  std::sregex_token_iterator it{str.begin(), str.end(), regex, -1};
  std::vector<std::string> words{it, {}};
  return words;
}

std::vector<std::string> str_split(const std::string &str, const std::regex &re) {
  std::sregex_token_iterator it{str.begin(), str.end(), re, -1};
  std::vector<std::string> words{it, {}};
  return words;
}

}