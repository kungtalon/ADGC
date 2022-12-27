//
// Created by kungtalon on 2022/12/26.
//

#ifndef ADGC_INCLUDE_AUTODIFF_LAYER_NORMALIZATION_H_
#define ADGC_INCLUDE_AUTODIFF_LAYER_NORMALIZATION_H_

#include "autodiff/layer/layer.h"

namespace auto_diff {
namespace layer {

class BatchNorm2D : public Layer {
 public:
  BatchNorm2D() {};
  BatchNorm2D(const size_t &num_channel,
              const double &epsilon = 1e-5,
              const double &momentum = 0.1,
              Graph *g = nullptr);
  ~BatchNorm2D() {};

  Parameter &get_weight();
  Parameter &get_bias();
  DTensor get_moving_mean();
  DTensor get_moving_var();
  Node &operator()(const Node &input);
  void assign_weight(const DTensor &value);
  void assign_bias(const DTensor &value);

 private:
  size_t num_channel_;
  functional::BatchNorm2D *bn_node_ptr_;
  double epsilon_, momentum_;

  void check_input(const Node &input);
};

}
}

#endif //ADGC_INCLUDE_AUTODIFF_LAYER_NORMALIZATION_H_
