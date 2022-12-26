//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_TENSOR_OPS_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_TENSOR_OPS_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class Add : public Node {
 public:
  Add() : Node(NodeType::ADG_ADD_TYPE) {};
  Add(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
      const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

// add a vector to a matrix with the same size in the last dim
class MatAddVec : public Node {
 public:
  MatAddVec() : Node(NodeType::ADG_MATADDVEC_TYPE) {};
  MatAddVec(Node *parent1_ptr, Node *parent2_ptr, const size_t &axis = SIZE_MAX, Graph *g = nullptr,
            const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  size_t axis_;
};

class VecDot : public Node {
 public:
  VecDot() : Node(NodeType::ADG_VECDOT_TYPE) {};
  VecDot(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatMul : public Node {
 public:
  MatMul() : Node(NodeType::ADG_MATMUL_TYPE) {};
  MatMul(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class MatSum : public Node {
 public:
  MatSum() : Node(NodeType::ADG_MATSUM_TYPE) {};
  MatSum(const std::vector<Node *> &parents, Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class PointMul : public Node {
 public:
  PointMul() : Node(NodeType::ADG_POINTMUL_TYPE) {};
  PointMul(Node *parent_ptr1, Node *parent_ptr2, Graph *g = nullptr,
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

}

}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_TENSOR_OPS_H_
