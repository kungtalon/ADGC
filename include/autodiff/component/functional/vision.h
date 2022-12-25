//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class Conv2D : public Node {
  Conv2D() : Node(NodeType::ADG_CONV2D_TYPE) {};
  Conv2D(Node *input_ptr,
         Parameter *kernel_ptr,
         const std::vector<size_t> &strides,
         Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  size_t out_c_, out_h_, out_w_;
  DTensor col_image_, col_kernel_;
  std::vector<size_t> strides_, kernel_shape_;

  void im2col(const DTensor &input);
  DTensor col2im(const DTensor &input);

};
}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_
