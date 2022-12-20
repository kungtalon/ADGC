#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include <algorithm>

#include "node.h"

namespace graph_component {
namespace ops {

class Add : public Node {
public:
  Add() : Node(NodeType::ADG_ADD_TYPE){};
  Add(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
      const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class VecDot : public Node {
public:
  VecDot() : Node(NodeType::ADG_VECDOT_TYPE){};
  VecDot(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatMul : public Node {
public:
  MatMul() : Node(NodeType::ADG_MATMUL_TYPE){};
  MatMul(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatSum : public Node {
public:
  MatSum() : Node(NodeType::ADG_MATSUM_TYPE){};
  MatSum(std::vector<Node *> parents, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

} // namespace ops

} // namespace graph_component

#endif