#ifndef ADGC_AUTODIFF_FUNCTIONAL_H_
#define ADGC_AUTODIFF_FUNCTIONAL_H_

#include "node.h"

namespace graph_component {
namespace functional {

class Sigmoid : public Node {
public:
  Sigmoid() : Node("sigmoid"){};
  Sigmoid(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class ReLU : public Node {
public:
  ReLU() : Node("relu"){};
  ReLU(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class CrossEntropyWithSoftMax : public Node {
public:
  CrossEntropyWithSoftMax() : Node("cross_entropy_softmax"){};
  CrossEntropyWithSoftMax(const std::vector<Node *> &parents);
  DTensor softmax(const DTensor &input);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace functional
} // namespace graph_component

#endif