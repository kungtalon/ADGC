#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include <algorithm>

#include "node.h"
#include "variable.h"

namespace auto_diff {
namespace ops {

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
  MatAddVec(Node *parent1_ptr, Node *parent2_ptr, Graph *g = nullptr,
            const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
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

class PointMul : public Node {
 public:
  PointMul() : Node(NodeType::ADG_POINTMUL_TYPE) {};
  PointMul(Node *parent_ptr1, Node *parent_ptr2, Graph *g = nullptr,
           const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
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

class Conv2D : public Node {
  Conv2D() : Node(NodeType::ADG_CONV2D_TYPE) {};
  Conv2D(Node *input_ptr,
         Parameter *kernel_ptr,
         const std::vector<size_t> &strides,
         Graph *g = nullptr,
         const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  size_t out_c_, out_h_, out_w_;
  DTensor col_image_, col_kernel_;
  std::vector<size_t> strides_, kernel_shape_;

  void im2col(const DTensor &input);
  DTensor col2im(const DTensor &input);
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