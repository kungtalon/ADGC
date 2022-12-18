#include "autodiff/node.h"

namespace graph_component {

Node::Node() : Node("unknown") {}

Node::Node(const std::string &type, const std::string &name, Graph *graph)
    : type_(type), value_(tensor::EMPTY), jacobi_(tensor::EMPTY) {
  if (graph == nullptr) {
    graph_ = Graph::get_instanceof_global_graph();
  } else {
    graph_ = graph;
  }

  name_ = graph_->add_node(this, type_, name);
}

Node::Node(const std::string &type, const std::vector<Node *> &parents,
           const std::string &name, Graph *graph)
    : type_(type), parents_(parents), value_(tensor::EMPTY),
      jacobi_(tensor::EMPTY) {
  if (graph == nullptr) {
    graph_ = Graph::get_instanceof_global_graph();
  } else {
    graph_ = graph;
  }

  for (auto parent_ptr : parents) {
    if (parent_ptr->get_graph() != graph_) {
      // parent nodes must belong to the same graph!
      throw adg_exception::MismatchRegisterdGraphError();
    }
    parent_ptr->add_children(this); // register this node to all its parents
  }
  name_ =
      graph_->add_node(this, type_, name); // register this node into the graph
};

void Node::forward() {
  for (auto parent_ptr : parents_) {
    if (parent_ptr->get_value() == tensor::EMPTY) {
      // if parent node didn't do forward propagation
      // let them do it first!
      parent_ptr->forward();
    }
    this->do_forward(); // compute is an abstract function
  }
}

DTensor Node::backward(Node *result) {
  if (jacobi_ == tensor::EMPTY) {
    if (this == result) {
      jacobi_ = tensor::Eye(get_value_size());
    } else {
      jacobi_ = tensor::Zeros({get_value_size(), get_value_size()});

      for (auto child_ptr : children_) {
        if (child_ptr->get_value() != tensor::EMPTY) {
          DTensor childs_backward = child_ptr->backward(result);
          DTensor childs_contrib =
              childs_backward.multiply(child_ptr->do_backward(this));
          jacobi_ = jacobi_.add(childs_contrib);
        }
      }
    }
  }
  return jacobi_;
}

void Node::clear_value(bool recursive) {
  value_ = tensor::EMPTY;

  if (recursive) {
    for (auto child_ptr : children_) {
      child_ptr->clear_value(recursive);
    }
  }
}

void Node::assign_value(const DTensor &value) {
  if (value.get_shape() != get_value_shape()) {
    throw adg_exception::MismatchTensorShapeError();
  }

  // all children's value should get re-computed
  clear_value(true);
  value_ = value;
}

// friend
void graph_reset_node_name(Node *node, const std::string &name, Graph *graph) {
  // only called by registred graph to re-assign the node name

  if (graph != node->graph_) {
    throw adg_exception::MismatchRegisterdGraphError();
  }

  node->name_ = name;
}

} // namespace graph_component