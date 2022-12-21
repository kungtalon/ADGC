#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include <algorithm>

#include "node.h"

namespace auto_diff {
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
  MatSum(const std::vector<Node *> &parents, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

Add &add(const Node &parent1, const Node &parent2, Graph *g = nullptr,
         const std::string &name = "");

VecDot &vecdot(const Node &parent1, const Node &parent2, Graph *g = nullptr,
               const std::string &name = "");

MatMul &matmul(const Node &parent1, const Node &parent2, Graph *g = nullptr,
               const std::string &name = "");

MatSum &matsum(const std::vector<Node *> &parents_ptr, Graph *g = nullptr,
               const std::string &name = "");

MatSum &matsum(const Node &parent_1, const Node &parent_2, Graph *g = nullptr,
               const std::string &name = "");

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               Graph *g = nullptr, const std::string &name = "");

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               const Node &parent_4, Graph *g = nullptr,
               const std::string &name = "");

} // namespace ops

} // namespace auto_diff

#endif