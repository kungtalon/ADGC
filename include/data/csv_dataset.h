//
// Created by kungtalon on 2022/12/23.
//

#ifndef ADGC_DATA_CSV_DATASET_H_
#define ADGC_DATA_CSV_DATASET_H_

#include "dataset.h"

namespace auto_diff {
namespace data {
class CsvDataset {
 public:
  CsvDataset(const std::string &path,
             const size_t &label_index = 0,  // which column index corresponds to label
             const size_t &batch_size = 8,
             const std::string &sep = ",",
             bool to_one_hot = false,
             bool shuffle = true,
             unsigned long shuffle_seed = 0);
  bool has_next() const;
  DataPair get_next();
  void reset_iterator();
  void reset_iterator(bool shuffle);
  void add_transform(std::function<void(tensor::Tensor<double> &)> trans_func);

  inline size_t get_label_count() const { return nlabels_; };
  inline size_t get_data_count() const { return labels_.size(); };

 private:
  bool shuffle_, to_one_hot_;
  size_t batch_size_, shuffle_seed_, nlabels_, column_size_, label_index_;
  std::ifstream file_reader_;
  std::string buffer_;
  std::vector<std::vector<double>> data_;
  std::vector<double> labels_;
  size_t iter_index_;
  std::vector<size_t> shuffle_indices_;
  std::vector<std::function<void(tensor::Tensor<double> &)>>
    transforms_; // transform functions should look like [](Tensor& ts) ( ts = ...; )

  void read_line(const std::string &sep);
  void do_shuffle(unsigned long seed = 0);
  void do_transform(tensor::Tensor<double> &);
};

}
}

#endif //ADGC_DATA_CSV_DATASET_H_
