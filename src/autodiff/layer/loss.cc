#include "autodiff/layer/loss.h"

namespace auto_diff {
namespace layer {

CrossEntropyWithSoftmax::CrossEntropyWithSoftmax(const std::string &reduction)
    : reduction_(reduction) {
  if (reduction != "mean" && reduction != "sum" && reduction != "none") {
    throw std::invalid_argument(
        "layer >> CrossEntropyWithSoftmax: invalid reduction method");
  }
}
}; // namespace layer
} // namespace auto_diff
