//
// Created by kungtalon on 2022/12/27.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_NORMALIZATION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_NORMALIZATION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class BatchNorm2D : public Node {
 public:
  BatchNorm2D() : Node(NodeType::ADG_BATCHNORM2D_TYPE) {};
  BatchNorm2D(Node *input_ptr,
              Parameter *gamma,
              Parameter *beta,
              const double &epsilon = 1e-5,
              const double &momentum = 0.1,
              Graph *g = nullptr,
              const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  DTensor moving_mean_, moving_var_;  // shape: [C]
  double epsilon_, momentum_;
};

}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_NORMALIZATION_H_
