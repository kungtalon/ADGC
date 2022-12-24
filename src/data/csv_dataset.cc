//
// Created by kungtalon on 2022/12/23.
//

#include "data/csv_dataset.h"

namespace auto_diff {
namespace data {

CsvDataset::CsvDataset(const std::string &path,
                       const size_t &label_index,
                       const size_t &batch_size,
                       const std::string &sep,
                       bool to_one_hot,
                       bool shuffle,
                       unsigned long shuffle_seed)
  : batch_size_(batch_size),
    shuffle_(shuffle),
    shuffle_seed_(shuffle_seed),
    label_index_(label_index),
    column_size_(0),
    nlabels_(0), to_one_hot_(to_one_hot), iter_index_(0) {
  file_reader_ = std::ifstream(path);

  std::regex blank("\\s*");
  try {
    while (std::getline(file_reader_, buffer_)) {
      if (std::regex_match(buffer_, blank)) {
        break;
      }
      read_line(sep);
    }
  } catch (const adg_exception::DatasetError &ex) {
    file_reader_.close();
    throw adg_exception::DatasetError(
      std::string("Error when reading from the dataset files... message: \n") + ex.what());
  }

  file_reader_.close();

  std::iota(shuffle_indices_.begin(), shuffle_indices_.end(), 0);
  if (shuffle_) {
    do_shuffle();
  }
  nlabels_ = std::unordered_set<double>(labels_.begin(), labels_.end()).size();
}

void CsvDataset::read_line(const std::string &sep) {
  auto splitted = utils::str_split(buffer_, sep);

  if (!column_size_) {
    column_size_ = splitted.size();
  }

  std::vector<double> row(column_size_ - 1);
  for (int ix = 0, iy = 0; ix < column_size_; ++ix) {
    if (ix == label_index_) {
      continue;
    }
    row[iy++] = std::stod(splitted[ix]);
  }
  data_.emplace_back(row);
  labels_.emplace_back(std::stod(splitted[label_index_]));
}

bool CsvDataset::has_next() const {
  return iter_index_ < labels_.size();
}

DataPair CsvDataset::get_next() {
  size_t ix = 0;
  std::vector<tensor::Tensor<double>> data_tensors;
  std::vector<tensor::Tensor<double>> label_tensors;
  while (iter_index_ != labels_.size() && ix++ < batch_size_) {
    if (to_one_hot_) {
      label_tensors.emplace_back(to_one_hot(nlabels_,
                                            tensor::Tensor<double>(std::vector<size_t>({1}), labels_[iter_index_])));
    } else {
      label_tensors.emplace_back(std::vector<size_t>({1}), labels_[iter_index_]);
    }

    data_tensors.emplace_back(std::vector<size_t>({1, column_size_ - 1}), data_[iter_index_++]);
  }

  return DataPair({tensor::Tensor<double>::concat(data_tensors, 0), tensor::Tensor<double>::concat(label_tensors, 0)});
}

void CsvDataset::reset_iterator() {
  if (shuffle_) {
    do_shuffle(shuffle_seed_);
  }
  iter_index_ = 0;
}

void CsvDataset::reset_iterator(bool shuffle) {
  if (shuffle) {
    do_shuffle(shuffle_seed_);
  }
  iter_index_ = 0;
}

void CsvDataset::add_transform(std::function<void(tensor::Tensor<double> &)> trans_func) {
  transforms_.emplace_back(trans_func);
}

void CsvDataset::do_shuffle(unsigned long seed) {
  auto rng = std::default_random_engine(seed);
  std::shuffle(std::begin(shuffle_indices_), std::end(shuffle_indices_), rng);
}

void CsvDataset::do_transform(tensor::Tensor<double> &ts) {
  for (auto tfunc : transforms_) {
    tfunc(ts);
  }
}

}
}