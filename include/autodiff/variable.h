#ifndef ADGC_AUTODIFF_VARIABLE_H_
#define ADGC_AUTODIFF_VARIABLE_H_

#include "node.h"

#include <string>
#include <vector>

namespace graph_component {

class Variable : public Node {
public:
  Variable() : Variable({1}, {}, "", true, true, nullptr){};
  Variable(const tensor::TensorShape &shape)
      : Variable(shape, {}, "", true, true, nullptr){};
  Variable(const tensor::TensorShape &shape, Graph *graph)
      : Variable(shape, {}, "", true, true, graph){};
  Variable(const tensor::TensorShape &shape, const std::vector<Node *> &parents,
           Graph *graph)
      : Variable(shape, parents, "", true, true, graph){};
  Variable(const tensor::TensorShape &shape, const std::vector<Node *> &parents,
           const std::string &name = "", const bool &random_init = true,
           const bool &trainable = true, Graph *graph = nullptr)
      : Node("variable", parents, name, graph) {
    value_ = DTensor(shape);
    if (random_init) {
      value_.normal_init(0., 0.001);
    }
  }

  void reset_value();

  void compute() override{};
  DTensor get_jacobi(Node *parent) override { return tensor::EMPTY; };

private:
  bool trainable_;
};

} // namespace graph_component

#endif