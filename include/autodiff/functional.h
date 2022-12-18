#ifndef ADGC_AUTODIFF_FUNCTIONAL_H_
#define ADGC_AUTODIFF_FUNCTIONAL_H_

#include "node.h"

namespace graph_component {
namespace functional {

class Sigmoid : public Node {
public:
  Sigmoid() : Node(NodeType::ADG_SIGMOID_TYPE){};
  Sigmoid(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class ReLU : public Node {
public:
  ReLU() : Node(NodeType::ADG_RELU_TYPE){};
  ReLU(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class CrossEntropyWithSoftMax : public Node {
public:
  CrossEntropyWithSoftMax() : Node(NodeType::ADG_CROSS_ENTROPY_SOFTMAX_TYPE){};
  CrossEntropyWithSoftMax(const std::vector<Node *> &parents);
  DTensor softmax(const DTensor &input);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace functional
} // namespace graph_component

#endif