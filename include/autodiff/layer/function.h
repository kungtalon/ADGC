#ifndef ADGC_AUTODIFF_LAYER_FUNCTION_H_
#define ADGC_AUTODIFF_LAYER_FUNCTION_H_

#include "layer.h"

namespace auto_diff {
namespace layer {

class ReLU : public Layer {
public:
  ReLU():Layer(LayerType::ADG_LAYER_RELU, nullptr){}
  ReLU(Graph* graph): Layer(LayerType::ADG_LAYER_RELU, graph) {};
  Node &operator()(const Node &input);
  Node &forward(const std::vector<std::string> &input_nodes) override;

private:
  Node *do_forward(const std::vector<Node *> &node_ptrs) override;
};

class Sigmoid : public Layer {
public:
  Sigmoid(): Layer(LayerType::ADG_LAYER_SIGMOID, nullptr){}
  Sigmoid(Graph* graph): Layer(LayerType::ADG_LAYER_SIGMOID, graph) {};
  Node &operator()(const Node &input);
  Node &forward(const std::vector<std::string> &input_nodes) override;

private:
  Node *do_forward(const std::vector<Node *> &node_ptrs) override;
};


} // namespace layer
} // namespace auto_diff


#endif