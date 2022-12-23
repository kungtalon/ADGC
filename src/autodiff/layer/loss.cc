#include "autodiff/layer/loss.h"

namespace auto_diff {
namespace layer {

CrossEntropyWithSoftmax::CrossEntropyWithSoftmax(const std::string &reduction, Graph *graph)
  : Layer(LayerType::ADG_LAYER_DENSE), reduction_(reduction) {
  if (reduction != "mean" && reduction != "sum" && reduction != "none") {
    throw std::invalid_argument(
      "layer >> CrossEntropyWithSoftmax: invalid reduction method");
  }
}

Node &operator()(const Node &input, const Variable &label) {

}

Node *do_forward() {

}

DTensor get_probs() {

}
}; // namespace layer
} // namespace auto_diff
