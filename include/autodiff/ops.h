#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include <algorithm>

#include "node.h"
namespace graph_component {
namespace ops {

class Add : public Node {
public:
  Add() : Node("add"){};
  Add(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class VecDot : public Node {
  VecDot() : Node("vecdot"){};
  VecDot(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatMul : public Node {
  MatMul() : Node("matmul"){};
  MatMul(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace ops

} // namespace graph_component

#endif