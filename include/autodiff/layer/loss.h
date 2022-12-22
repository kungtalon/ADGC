#ifndef ADGC_AUTODIFF_LAYER_LOSS_H_
#define ADGC_AUTODIFF_LAYER_LOSS_H_

#include "layer.h"

namespace auto_diff {
namespace layer {

class CrossEntropyWithSoftmax {
public:
  CrossEntropyWithSoftmax(){};
  CrossEntropyWithSoftmax(const std::string& reduction="mean");
  ~CrossEntropyWithSoftmax(){};
  
private:
   std::string reduction_;
};





} // namespace layer
} // namespace auto_diff

#include "autodiff/layer/dense.h"

#endif