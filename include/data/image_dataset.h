//
// Created by kungtalon on 12/22/22.
//

#ifndef ADGC_INCLUDE_DATA_IMAGE_DATASET_H_
#define ADGC_INCLUDE_DATA_IMAGE_DATASET_H_

#define STB_IMAGE_IMPLEMENTATION

#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <functional>
#include <unordered_set>

#include "stb_image.h"
#include "tensor/tensor.h"

namespace auto_diff {

typedef std::pair<tensor::Tensor<double>, tensor::Tensor<double>> DataPair;

namespace data {

namespace fs = std::filesystem;

struct Image {
  size_t width_, height_, channels_;
  unsigned char *data_;

  ~Image() {
    if (data_ != nullptr) {
      stbi_image_free(data_);
      data_ = nullptr;
    }
  }
};

class ImageDataset {
 public:
  ImageDataset(const std::string &root,
               const std::string &list_txt,
               const size_t &batch_size,
               bool shuffle = true,
               unsigned long shuffle_seed = 0);
  Image *read_image(const std::string &file_name);
  bool has_next() const;
  DataPair get_next();
  tensor::Tensor<double> image_to_tensor(Image *img);
  void reset_iterator();
  void reset_iterator(bool shuffle);
  void add_transform(std::function<void(tensor::Tensor<double> &)> trans_func);

  inline size_t get_label_count() const { return nlabels_; };

 private:
  bool shuffle_;
  size_t batch_size_, shuffle_seed_, nlabels_;
  std::vector<std::string>::iterator image_iter_;
  std::vector<std::string> img_file_list_;
  std::vector<double> labels_;
  std::vector<double>::iterator label_iter_;
  std::vector<std::function<void(tensor::Tensor<double> &)>>
    transforms_; // transform functions should look like [](Tensor& ts) ( ts = ...; )

  void do_shuffle(unsigned long seed = 0);
  void do_transform(tensor::Tensor<double> &);
};

}

}
#endif //ADGC_INCLUDE_DATA_IMAGE_DATASET_H_
