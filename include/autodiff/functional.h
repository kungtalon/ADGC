#ifndef ADGC_AUTODIFF_FUNCTIONAL_H_
#define ADGC_AUTODIFF_FUNCTIONAL_H_

#include "node.h"
#include "variable.h"

namespace graph_component {
namespace functional {

class Sigmoid : public Node {
public:
  Sigmoid() : Node(NodeType::ADG_SIGMOID_TYPE){};
  Sigmoid(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class ReLU : public Node {
public:
  ReLU() : Node(NodeType::ADG_RELU_TYPE){};
  ReLU(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class CrossEntropyWithSoftMax : public Node {
public:
  CrossEntropyWithSoftMax() : Node(NodeType::ADG_CROSS_ENTROPY_SOFTMAX_TYPE){};
  CrossEntropyWithSoftMax(Node *parent_ptr, Variable *labels_ptr,
                          Graph *g = nullptr, const std::string &name = "");
  static DTensor softmax(const DTensor &input);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

private:
  static inline double epsilon_ = 1e-9;
  DTensor probs_;
  DTensor neg_log_probs_;
};

class ReduceSum : public Node {
public:
  ReduceSum() : Node(NodeType::ADG_REDUCE_SUM_TYPE){};
  ReduceSum(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace functional
} // namespace graph_component

#endif