#include "autodiff/variable.h"

namespace graph_component {

Variable::Variable() : Variable({1}, {}, "", true, true, nullptr){};

Variable::Variable(const tensor::TensorShape &shape)
    : Variable(shape, {}, "", true, true, nullptr){};

Variable::Variable(const tensor::TensorShape &shape, Graph *graph)
    : Variable(shape, {}, "", true, true, graph){};

Variable::Variable(const tensor::TensorShape &shape, const bool &trainable,
                   Graph *graph)
    : Variable(shape, {}, "", true, trainable, graph){};

Variable::Variable(const tensor::TensorShape &shape,
                   const std::vector<Node *> &parents, Graph *graph)
    : Variable(shape, parents, "", true, true, graph){};

Variable::Variable(const tensor::TensorShape &shape,
                   const std::vector<Node *> &parents, const std::string &name,
                   const bool &random_init, const bool &trainable, Graph *graph)
    : Node(NodeType::ADG_VARIABLE_TYPE, parents, name, graph),
      trainable_(trainable) {
  unique_ptr_->assign_value(DTensor(shape), false);
  if (random_init) {
    get_value().normal_init(0., 0.001);
  }
}

DTensor Variable::do_backward(Node *parent_ptr) {
  return tensor::EMPTY;
} // do nothing

Parameter::Parameter(){};

Parameter::Parameter(const tensor::TensorShape &shape, const std::string &name,
                     Graph *graph)
    : Node(NodeType::ADG_PARAMETER_TYPE, name, graph), trainable_(true) {
  unique_ptr_->assign_value(DTensor(shape), false);
  get_value().normal_init(0., 0.001);
}

DTensor Parameter::do_backward(Node *parent) { return tensor::EMPTY; }

} // namespace graph_component
