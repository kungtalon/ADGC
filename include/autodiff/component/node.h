#ifndef ADGC_AUTODIFF_NODE_H_
#define ADGC_AUTODIFF_NODE_H_

#include <string>
#include <vector>

#include "assert.h"
#include "autodiff/consts.h"
#include "tensor/tensor.h"
#include "tensor/extension.h"
#include "utils/utils.h"

namespace auto_diff {

class Graph;

typedef tensor::Tensor<double> DTensor;

class Node {
 public:
  Node();
  Node(const std::string &type, const std::string &name = "",
       Graph *graph = nullptr);
  Node(const std::string &type, const std::vector<Node *> &parents,
       const std::string &name = "", Graph *graph = nullptr);
  Node(const std::string &type, std::vector<Node> &parents,
       const std::string &name = "", Graph *graph = nullptr);
  Node(const Node &other);
  Node(const Node &&other);
  Node &operator=(const Node &other);
  virtual ~Node() {};

  void clear_value(bool recursive = true);
  void assign_value(const DTensor &value, bool check_shape = true);
  DTensor get_grad(bool reshaped = true) const;

  virtual void forward();
  virtual DTensor backward(Node *result);

  inline std::string get_type() const { return type_; }
  inline std::string get_name() const { return name_; }
  inline std::string get_full_name() const { return type_ + "_" + name_; }
  inline Node *get_ptr() { return unique_ptr_; }
  inline const Graph *get_graph() const { return unique_ptr_->graph_; }
  inline std::vector<Node *> get_children() const {
    return unique_ptr_->children_;
  }
  inline std::vector<Node *> get_parents() const {
    return unique_ptr_->parents_;
  }
  inline DTensor get_value() const { return unique_ptr_->value_; }
  inline bool is_value_empty() const { return unique_ptr_->empty_value_; };
  inline bool is_grad_empty() const { return unique_ptr_->empty_jacobi_; };
  inline size_t get_value_size() const {
    return unique_ptr_->value_.get_size();
  }
  inline tensor::TensorShape get_value_shape() const {
    return unique_ptr_->value_.get_shape();
  }
  inline size_t get_value_dim() const {
    return unique_ptr_->value_.get_dim();
  }

  // inline void set_graph(Graph *graph) { graph_ = graph; }
  inline void clear_jacobi() {
    if (!unique_ptr_->is_grad_empty()) {
      unique_ptr_->jacobi_ = tensor::EMPTY;
      unique_ptr_->empty_jacobi_ = true;
    }
  }
  inline void add_children(Node *child) {
    unique_ptr_->children_.push_back(child->unique_ptr_);
  }
  inline void add_parent(Node *parent) {
    assert(parent->get_graph() == graph_);
    unique_ptr_->parents_.push_back(parent->unique_ptr_);
  }

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
  Node *unique_ptr_; // use this as the reference to the unique node pointer

  virtual void do_forward() = 0;                 // compute value
  virtual DTensor do_backward(Node *parent) = 0; // compute jacobian
};

} // namespace auto_diff

#include "autodiff/graph.h"

#endif