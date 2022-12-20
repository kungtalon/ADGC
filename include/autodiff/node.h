#ifndef ADGC_AUTODIFF_NODE_H_
#define ADGC_AUTODIFF_NODE_H_

#include <string>
#include <vector>

#include "consts.h"
#include "tensor/tensor.h"
#include "utils/utils.h"

namespace graph_component {

class Graph; // forward

typedef tensor::Tensor<double> DTensor;

class Node {
public:
  Node();
  Node(const std::string &type, const std::string &name = "",
       Graph *graph = nullptr);
  Node(const std::string &type, const std::vector<Node *> &parents,
       const std::string &name = "", Graph *graph = nullptr);
  void clear_value(bool recursive = true);
  void assign_value(const DTensor &value);

  virtual void forward();
  virtual DTensor backward(Node *result);

  inline std::string get_type() const { return type_; }
  inline std::string get_name() const { return name_; }
  inline std::string get_full_name() const { return type_ + "_" + name_; }
  inline Graph *get_graph() { return graph_; }
  inline std::vector<Node *> get_children() const { return children_; }
  inline std::vector<Node *> get_parents() const { return parents_; }
  inline DTensor get_value() const { return value_; }
  inline DTensor get_grad(bool reshaped = true) const {
    if (reshaped) {
      DTensor jacobi_cp = jacobi_;
      jacobi_cp.reshape(value_.get_shape());
      return jacobi_cp;
    }
    return jacobi_;
  }
  inline bool is_value_empty() const { return empty_value_; };
  inline bool is_grad_empty() const { return empty_jacobi_; };
  inline size_t get_value_size() const { return value_.get_size(); }
  inline tensor::TensorShape get_value_shape() const {
    return value_.get_shape();
  }

  // inline void set_graph(Graph *graph) { graph_ = graph; }
  inline void clear_jacobi() {
    if (!empty_jacobi_) {
      jacobi_ = tensor::EMPTY;
      empty_jacobi_ = true;
    }
  }
  inline void add_children(Node *child) { children_.push_back(child); }

  friend void graph_reset_node_name(Node *node, const std::string &name,
                                    Graph *graph);

protected:
  std::string type_;
  std::string name_;
  std::vector<Node *> parents_;
  std::vector<Node *> children_;
  DTensor value_;
  bool empty_value_;
  DTensor jacobi_;
  bool empty_jacobi_;
  Graph *graph_;

  virtual void do_forward() = 0;                 // compute value
  virtual DTensor do_backward(Node *parent) = 0; // compute jacobian
};

} // namespace graph_component

#include "graph.h"

#endif