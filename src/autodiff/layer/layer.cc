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

void Layer::freeze() {
  for (auto param_ptr : params_ptr_list_) {
    param_ptr->set_trainable(false);
  }
}

void Layer::add_param(Parameter *param_ptr) {
  params_ptr_list_.emplace_back(param_ptr);
}

std::vector<Parameter *> Layer::get_param_ptr_list() const {
  return params_ptr_list_;
}

} // namespace layer
} // namespace auto_diff
