//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class Conv2D : public Node {
 public:
  Conv2D() : Node(NodeType::ADG_CONV2D_TYPE) {};
  Conv2D(Node *input_ptr,
         Parameter *kernel_ptr,
         const size_t &stride,
         Graph *g = nullptr,
         const std::string &name = "");
  Conv2D(Node *input_ptr,
         Parameter *kernel_ptr,
         const std::array<size_t, 2> &strides,
         Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  size_t out_c_, out_h_, out_w_, residual_h_, residual_w_;
  DTensor col_image_, col_kernel_;
  std::vector<size_t> kernel_shape_;
  std::array<size_t, 2> strides_;

  DTensor im2col_hwc(const DTensor &input,
                     const size_t &kh,
                     const size_t &kw,
                     const size_t &sh,
                     const size_t &sw);
  DTensor im2col_chw(const DTensor &input,
                     const size_t &kh,
                     const size_t &kw,
                     const size_t &sh,
                     const size_t &sw);

};




// functions

}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_
