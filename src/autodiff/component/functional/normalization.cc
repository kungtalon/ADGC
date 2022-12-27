//
// Created by kungtalon on 2022/12/27.
//


#include "autodiff/component/functional/normalization.h"

namespace auto_diff {
namespace functional {

BatchNorm2D::BatchNorm2D(Node *input_ptr,
                         Parameter *gamma,
                         Parameter *beta,
                         const double &epsilon,
                         const double &momentum,
                         Graph *g,
                         const std::string &name)
  : Node(NodeType::ADG_MATMUL_TYPE, {input_ptr, gamma, beta}, name, g),
    momentum_(momentum),
    epsilon_(epsilon) {

}

void BatchNorm2D::do_forward() {
  DTensor input_tensor = parents_[0]->get_value();

  size_t value_size = input_tensor.get_size();
  size_t size_bhw = value_size / input_tensor.get_shape(1);
  DTensor sample_mean = input_tensor.mean(3).mean(2).mean(0).multiply(1. / size_bhw);
}

DTensor BatchNorm2D::do_backward(Node *parent_ptr) {
  return DTensor();
}

}
}