#include "autodiff/node.h"

namespace graph_component {

Node::Node() : Node(NodeType::ADG_UNKNOWN_TYPE) {}

Node::Node(const std::string &type, const std::string &name, Graph *graph)
    : type_(type), empty_jacobi_(true), empty_value_(true) {
  value_ = tensor::EMPTY;
  jacobi_ = tensor::EMPTY;
  if (graph == nullptr) {
    graph_ = Graph::get_instanceof_global_graph();
  } else {
    graph_ = graph;
  }

  name_ = graph_->add_node(this, type_, name);
}

Node::Node(const std::string &type, const std::vector<Node *> &parents,
           const std::string &name, Graph *graph)
    : type_(type), parents_(parents), empty_jacobi_(true), empty_value_(true) {
  value_ = tensor::EMPTY;
  jacobi_ = tensor::EMPTY;
  if (graph == nullptr) {
    graph_ = Graph::get_instanceof_global_graph();
  } else {
    graph_ = graph;
  }

  for (auto parent_ptr : parents) {
    if (parent_ptr->get_graph() != graph_) {
      // parent nodes must belong to the same graph!
      throw adg_exception::MismatchRegisterdGraphError(
          "Different graphs for a node " + type + " and parent node " +
          parent_ptr->get_full_name());
    }
    parent_ptr->add_children(this); // register this node to all its parents
  }
  name_ =
      graph_->add_node(this, type_, name); // register this node into the graph
};

void Node::forward() {
  for (auto parent_ptr : parents_) {
    if (parent_ptr->is_value_empty()) {
      // if parent node didn't do forward propagation
      // let them do it first!
      parent_ptr->forward();
    }
    this->do_forward(); // compute is an abstract function
  }
  empty_value_ = false;
}

DTensor Node::backward(Node *result) {
  if (result->get_value_size() != 1) {
    throw adg_exception::GradError("Target is not scalar!");
  }
  if (is_grad_empty()) {
    if (this == result) {
      jacobi_ = tensor::Ones({get_value_size(), 1});
    } else {
      jacobi_ = tensor::Zeros({get_value_size(), 1});

#if ADG_DEBUG_GLOABL_BOOL_
      for (auto child_ptr : children_) {
        if (!child_ptr->is_value_empty()) {
          DTensor childs_backward, childs_contrib;
          try {
            childs_backward = child_ptr->backward(result);
          } catch (const adg_exception::AutoDiffGraphException &ex) {
            throw adg_exception::TestingDebugException(
                "Getting tensor exception when backward...\nParent is " +
                get_full_name() + "\nChild is " + child_ptr->get_full_name() +
                "\nError msg: " + ex.what());
          }
          // childs_backward shape: [child_size, 1]
          // do_backward shape: [parent_size, child_size]
          // result shape: [parent_size, 1]
          try {
            childs_contrib = child_ptr->do_backward(this).dot(childs_backward);
          } catch (const adg_exception::AutoDiffGraphException &ex) {
            throw adg_exception::TestingDebugException(
                "Tensor Exception when dot jacobi in node " + get_full_name() +
                ", exception message: \n" + ex.what());
          }

          try {
            jacobi_ = jacobi_.add(childs_contrib);
          } catch (const adg_exception::AutoDiffGraphException &ex) {
            throw adg_exception::TestingDebugException(
                "Getting tensor exception when adding jacobi...\nParent is " +
                get_full_name() + "\nChild is " + child_ptr->get_full_name() +
                "\nError msg: " + ex.what());
          }
        }
      }
#else
      for (auto child_ptr : children_) {
        if (!child_ptr->is_value_empty()) {
          DTensor childs_backward, childs_contrib;
          childs_backward = child_ptr->backward(result);
          childs_contrib = child_ptr->do_backward(this).dot(childs_backward);
          jacobi_ = jacobi_.add(childs_contrib);
        }
      }
#endif
    }
  }

  empty_jacobi_ = false;
  return jacobi_;
}

void Node::clear_value(bool recursive) {
  value_ = tensor::EMPTY;
  empty_value_ = true;

  if (recursive) {
    for (auto child_ptr : children_) {
      child_ptr->clear_value(recursive);
    }
  }
}

void Node::assign_value(const DTensor &value) {
  if (value.get_shape() != get_value_shape()) {
    throw adg_exception::MismatchTensorShapeError(
        "MismatchTensorShapeError >> Node::assign_value get different value "
        "shapes: " +
        utils::vector_to_str(value.get_shape()) + " and " +
        utils::vector_to_str(get_value_shape()));
  }

  // all children's value should get re-computed
  clear_value(true);
  value_ = value;
  empty_value_ = false;
}

// not used
// friend
void graph_reset_node_name(Node *node, const std::string &name, Graph *graph) {
  // only called by registred graph to re-assign the node name

  if (graph != node->graph_) {
    throw adg_exception::MismatchRegisterdGraphError();
  }

  node->name_ = name;
}

} // namespace graph_component