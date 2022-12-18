#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include <algorithm>

#include "node.h"

namespace graph_component {
namespace ops {

class Add : public Node {
public:
  Add() : Node(NodeType::ADG_ADD_TYPE){};
  Add(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class VecDot : public Node {
  VecDot() : Node(NodeType::ADG_VECDOT_TYPE){};
  VecDot(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatMul : public Node {
  MatMul() : Node(NodeType::ADG_MATMUL_TYPE){};
  MatMul(const std::vector<Node *> &parents);
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace ops

} // namespace graph_component

#endif