#include "autodiff/layer/loss.h"

namespace auto_diff {
namespace layer {

CrossEntropyWithSoftmax::CrossEntropyWithSoftmax(const std::string &reduction, Graph *graph)
  : Layer(LayerType::ADG_LAYER_DENSE), reduction_(reduction), output_ptr_(nullptr) {
  if (reduction != "mean" && reduction != "sum" && reduction != "none") {
    throw std::invalid_argument(
      "layer >> CrossEntropyWithSoftmax: invalid reduction method");
  }
}

Node &CrossEntropyWithSoftmax::operator()(const Node &input, const Variable &label) {
  functional::CrossEntropyWithSoftMax &output = functional::cross_entropy_with_softmax(input, label);
  output_ptr_ = &output;
  return output;
}

DTensor CrossEntropyWithSoftmax::get_probs() {
  if (output_ptr_ == nullptr) {
    throw adg_exception::NodeValueError("Not set up node yet...");
  }
  return output_ptr_->get_probs();
}
}; // namespace layer
} // namespace auto_diff
