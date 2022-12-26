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
         const std::vector<size_t> &strides,
         Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

  inline DTensor get_im2col() {
    if (this != unique_ptr_) {
      auto *real_ptr = dynamic_cast<Conv2D *>(unique_ptr_);
      return real_ptr->col_image_;
    }
    return col_image_;
  };

 private:
  size_t out_c_, out_h_, out_w_;
  DTensor col_image_, col_kernel_;
  std::vector<size_t> strides_, kernel_shape_;

  void im2col_hwc(const DTensor &input,
                  DTensor &output,
                  const size_t &kh,
                  const size_t &kw,
                  const size_t &sh,
                  const size_t &sw);
  void im2col_chw(const DTensor &input,
                  DTensor &output,
                  const size_t &kh,
                  const size_t &kw,
                  const size_t &sh,
                  const size_t &sw);

};
}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_VISION_H_
