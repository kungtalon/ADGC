#ifndef ADGC_AUTODIFF_OPTIMIZER_GRADIENT_DESCENT_H_
#define ADGC_AUTODIFF_OPTIMIZER_GRADIENT_DESCENT_H_

#include "optimizer.h"

namespace auto_diff {
namespace optimizer {

class GradientDescent : public Optimizer {
 public:
  GradientDescent() {};
  GradientDescent(const Node &target,
                  const double &learning_rate = 0.01, Graph *graph = nullptr);

 private:
  void update(); // update the gradient to parameters
};
} // namespace optimizer
} // namespace auto_diff

#endif