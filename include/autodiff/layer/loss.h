#ifndef ADGC_AUTODIFF_LAYER_LOSS_H_
#define ADGC_AUTODIFF_LAYER_LOSS_H_

#include "layer.h"

namespace auto_diff {
namespace layer {

class CrossEntropyWithSoftmax : public Layer {
 public:
  CrossEntropyWithSoftmax() {};
  CrossEntropyWithSoftmax(const std::string &reduction = "mean", Graph *graph = nullptr);
  ~CrossEntropyWithSoftmax() {};

  DTensor get_probs();
  Node &operator()(const Node &input, const Variable &label);

 private:
  std::string reduction_;
  DTensor probs_;

};

} // namespace layer
} // namespace auto_diff

#include "autodiff/layer/dense.h"

#endif