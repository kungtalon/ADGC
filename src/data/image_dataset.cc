//
// Created by kungtalon on 12/22/22.
//
#include "data/image_dataset.h"

namespace auto_diff {
namespace data {

ImageDataset::ImageDataset(const std::string &root,
                           const std::string &list_txt,
                           const size_t &batch_size,
                           bool shuffle,
                           unsigned long shuffle_seed)
  : batch_size_(batch_size), shuffle_(shuffle), shuffle_seed_(shuffle_seed) {
  utils::read_lines_from_file(list_txt.c_str(), img_file_list_);
  if (shuffle) {
    do_shuffle(shuffle_seed);
  }

  std::unordered_set<double> labels_set;

  fs::path root_path(root);
  std::vector<std::string> splitted;
  std::regex sep_re("[\\t]");
  try {
    for (std::string &line : img_file_list_) {
      splitted = utils::str_split(line, sep_re);
      if (splitted.size() != 2) {
        throw adg_exception::DatasetError("The list txt file should contain two columns, file name and label");
      }

      fs::path file(splitted[0]);
      line = (root_path / file).string();

      try {
        double new_label = std::stod(splitted[1]);
        labels_.emplace_back(new_label);
        labels_set.insert(new_label);
      } catch (const std::exception &ex) {
        throw adg_exception::DatasetError("Failed to read labels correctly, the original line is:\n " + line);
      }
    }
  } catch (const adg_exception::DatasetError &ex) {
    throw adg_exception::DatasetError(
      std::string("Error when reading from the dataset files... message: \n") + ex.what());
  }

  assert(img_file_list_.size() == labels_.size());

  image_iter_ = img_file_list_.begin();
  label_iter_ = labels_.begin();
  nlabels_ = labels_set.size();
}

void ImageDataset::do_shuffle(unsigned long seed) {
  auto rng = std::default_random_engine(seed);
  std::shuffle(std::begin(img_file_list_), std::end(img_file_list_), rng);
}

bool ImageDataset::has_next() const {
  return image_iter_ != img_file_list_.end();
}

DataPair ImageDataset::get_next() {
  size_t counter = 0;
  std::vector<tensor::Tensor<double>> feature_tensors;
  std::vector<tensor::Tensor<double>> label_tensors;

  while (image_iter_ != img_file_list_.end() && counter++ < batch_size_) {
    feature_tensors.emplace_back(image_to_tensor(read_image(*(image_iter_++))));  // shape {1, h, w, c}
    label_tensors.emplace_back(std::vector<size_t>(1, 1), *(label_iter_++)); // shape {1}
  }

  return DataPair({tensor::Tensor<double>::concat(feature_tensors, 0),
                   tensor::Tensor<double>::concat(label_tensors, 0)});
}

Image *ImageDataset::read_image(const std::string &file_name) {
  struct Image *result = new Image;
  int width, height, channels;
  result->data_ = stbi_load(file_name.c_str(), &width, &height, &channels, 0);
  result->width_ = width;
  result->height_ = height;
  result->channels_ = channels;
  return result;
}

tensor::Tensor<double> ImageDataset::image_to_tensor(Image *img) {
  // tensor should have dim of [1, H, W, C] for convenient concat...
  size_t size = img->height_ * img->width_ * img->channels_;
  std::vector<double> result_data(img->data_, img->data_ + size);
  tensor::Tensor<double> result({1, img->height_, img->width_, img->channels_}, std::move(result_data));
  if (!transforms_.empty()) {
    do_transform(result);
  }
  return result;
}

void ImageDataset::reset_iterator() {
  if (shuffle_) {
    do_shuffle(shuffle_seed_);
  }
  image_iter_ = img_file_list_.begin();
  label_iter_ = labels_.begin();
}

void ImageDataset::reset_iterator(bool shuffle) {
  if (shuffle) {
    do_shuffle(shuffle_seed_);
  }
  image_iter_ = img_file_list_.begin();
  label_iter_ = labels_.begin();
}

void ImageDataset::add_transform(std::function<void(tensor::Tensor<double> &)> trans_func) {
  transforms_.emplace_back(trans_func);
}

void ImageDataset::do_transform(tensor::Tensor<double> &ts) {
  for (auto tfunc : transforms_) {
    tfunc(ts);
  }
}

}
}

