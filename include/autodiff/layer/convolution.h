//
// Created by kungtalon on 2022/12/26.
//

#ifndef ADGC_INCLUDE_AUTODIFF_LAYER_CONVOLUTION_H_
#define ADGC_INCLUDE_AUTODIFF_LAYER_CONVOLUTION_H_

#include "layer.h"
namespace auto_diff {

namespace layer {

class Conv2D : public Layer {
 public:
  Conv2D() {};
  Conv2D(const size_t &input_channel,
         const size_t &output_channel,
         const std::array<size_t, 2> &kernel_size,
         const std::array<size_t, 2> &stride,
         const std::array<size_t, 2> &padding,
         const std::string &activation = "relu",
         bool use_bias = true,
         Graph *graph = nullptr);
  Conv2D(const size_t &input_channel,
         const size_t &output_channel,
         const std::array<size_t, 2> &kernel_size,
         const std::array<size_t, 2> &stride,
         const std::string_view &padding = "VALID",
         const std::string &activation = "relu",
         bool use_bias = true,
         Graph *graph = nullptr);
  ~Conv2D() {};

  Parameter &get_weight();
  Parameter &get_bias();
  Node &operator()(const Node &input);
  void assign_weight(const DTensor &value);
  void assign_bias(const DTensor &value);

 private:
  size_t input_channel_, output_channel_;
  std::array<size_t, 4> padding_;
  std::array<size_t, 2> stride_;

  void check_input(const Node &input);
};
}
}
#endif //ADGC_INCLUDE_AUTODIFF_LAYER_CONVOLUTION_H_
