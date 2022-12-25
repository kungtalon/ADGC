//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_OPS_MANIPULATION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_OPS_MANIPULATION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class Reshape : public Node {
 public:
  Reshape() : Node(NodeType::ADG_POINTMUL_TYPE) {};
  Reshape(Node *parent_ptr, const tensor::TensorShape &shape, Graph *g = nullptr,
          const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  tensor::TensorShape new_shape_;
};

class Pad2D : public Node {
 public:
  Pad2D() : Node(NodeType::ADG_PAD2D_TYPE) {};
  Pad2D(Node *parent_ptr, const std::vector<std::pair<size_t, size_t>> &padding, const double &value = 0,
        Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
 private :
  double pad_value_;
  std::vector<std::pair<size_t, size_t>> padding_;

};

}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_OPS_MANIPULATION_H_
