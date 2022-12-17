#include "autodiff/node.h"

namespace graph_component {
Node::Node(NodePtrList parents)
    : Node(parents, "unnamed_node_" + std::to_string(node_idx++)) {}

Node::Node(NodePtrList parents, std::string name)
    : parents_(parents), name_(name) {
  this->graph_ = Graph::get_instanceof_global_graph();
  for (Node *parent : parents) {
    parent->add_children(this); // register this node to all its parents
  }
  graph_->add_node(this); // register this node into the graph
};

void Node::forward() {
  for (Node *parent : parents_) {
    if (parent->value_ == nullptr) {
      // if parent node didn't do forward propagation
      // let them do it first!
      parent->forward();
    }
    this->compute(); // compute is an abstract function
  }
}

void Node::backward() {
  if (jacobi_ == nullptr) {
  }
}

void Node::clear_jacobi() { jacobi_ = nullptr; }
void Node::clear_value(bool recursive) {}
void Node::add_children(Node *child) { children_.push_back(child); }
} // namespace graph_component