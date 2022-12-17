#ifndef ADGC_AUTODIFF_NODE_H_
#define AGDC_AUTODIFF_NODE_H_

#include <string>
#include <vector>

#include "graph.h"
#include "tensor.h"

namespace graph_component {

typedef std::vector<Node *> NodePtrList;

class Node {
protected:
  std::string type_;
  std::string name_;
  NodePtrList parents_;
  NodePtrList children_;
  tensor::Tensor<double> *value_;
  tensor::Tensor<double> *jacobi_;
  Graph *graph_;
  static size_t node_idx;

public:
  Node(NodePtrList parents);
  Node(NodePtrList parents, std::string name);
  void clear_jacobi();
  void clear_value(bool recursive);
  void add_children(Node *child);

  virtual void forward();
  virtual void backward();
  virtual void compute();
  virtual void get_jacobi();

  inline void set_value(tensor::Tensor<double> *value) { value_ = value; };
  inline tensor::Tensor<double> *get_value() { return value_; };
  inline void clear_jacobi() { jacobi_ = 0; };
  inline void set_graph(Graph *graph) { graph_ = graph; }
  inline Graph *get_graph() { return graph_; };
  inline NodePtrList &get_children() { return children_; };
  inline NodePtrList &get_parents() { return parents_; };
};
} // namespace graph_component

#endif