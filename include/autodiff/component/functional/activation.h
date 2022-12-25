//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_ACTIVATION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_ACTIVATION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class Sigmoid : public Node {
 public:
  Sigmoid() : Node(NodeType::ADG_SIGMOID_TYPE) {};
  Sigmoid(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class ReLU : public Node {
 public:
  ReLU() : Node(NodeType::ADG_RELU_TYPE) {};
  ReLU(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

Sigmoid &sigmoid(const Node &parent, Graph *g = nullptr,
                 const std::string &name = "");

ReLU &relu(const Node &parent, Graph *g = nullptr,
           const std::string &name = "");

}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_ACTIVATION_H_
