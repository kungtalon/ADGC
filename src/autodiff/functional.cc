#include "autodiff/functional.h"

namespace graph_component {

namespace functional {

Sigmoid &sigmoid(const Node &parent, Graph *g, const std::string &name) {
  Sigmoid *node_ptr =
      new Sigmoid(Graph::get_ptr_of(parent.get_full_name(), g), g, name);
  return *node_ptr;
}

ReLU &relu(const Node &parent, Graph *g, const std::string &name) {
  ReLU *node_ptr =
      new ReLU(Graph::get_ptr_of(parent.get_full_name(), g), g, name);
  return *node_ptr;
}

CrossEntropyWithSoftMax &cross_entropy_with_softmax(const Node &parent,
                                                    const Variable &labels,
                                                    Graph *g,
                                                    const std::string &name) {
  Node *parent_ptr = Graph::get_ptr_of(parent.get_full_name(), g);
  Variable *label_ptr =
      dynamic_cast<Variable *>(Graph::get_ptr_of(labels.get_full_name(), g));
  CrossEntropyWithSoftMax *node_ptr =
      new CrossEntropyWithSoftMax(parent_ptr, label_ptr, g, name);
  return *node_ptr;
}

ReduceSum &reduce_sum(const Node &parent, Graph *g, const std::string &name) {
  ReduceSum *node_ptr =
      new ReduceSum(Graph::get_ptr_of(parent.get_full_name(), g), g, name);
  return *node_ptr;
}

} // namespace functional

} // namespace graph_component