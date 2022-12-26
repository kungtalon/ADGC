#include "autodiff/layer/layer.h"

namespace auto_diff {
namespace layer {

Layer::Layer(const std::string &layer_type, Graph *g) {
  if (g == nullptr) {
    g = Graph::get_instanceof_global_graph();
  }
  graph_ = g;

  size_t id = g->counter_increment(layer_type);
  layer_name_ = layer_type + "_" + std::to_string(id);
} // namespace layer


void Layer::set_config(const std::string &key, const std::string &value) {
  configs_[key] = value;
}

std::string Layer::get_config(const std::string &key) {
  auto lookup_iter = configs_.find(key);
  if (lookup_iter == configs_.end()) {
    return "";
  } else {
    return lookup_iter->second;
  }
}

void Layer::freeze() {
  for (auto param_ptr : params_ptr_list_) {
    param_ptr->set_trainable(false);
  }
}

void Layer::add_param(Parameter *param_ptr) {
  params_ptr_list_.emplace_back(param_ptr);
}

Node *Layer::use_activation(Node *input) {
  std::string activation = get_config("activation");
  Node *output = input;
  if (activation == "relu") {
    output = new functional::ReLU(input, graph_, layer_name_ + "_relu");
  } else if (activation == "sigmoid") {
    output = new functional::Sigmoid(input, graph_, layer_name_ + "_sigmoid");
  } else if (activation == "none") {
    // do nothing
  } else {
    throw std::invalid_argument("layer " + layer_name_ +
      " receives invalid activation");
  }
  return output;
}

std::vector<Parameter *> Layer::get_param_ptr_list() const {
  return params_ptr_list_;
}

} // namespace layer
} // namespace auto_diff
