#ifndef ADGC_AUTODIFF_OPTIMIZER_GRADIENT_DESCENT_H_
#define ADGC_AUTODIFF_OPTIMIZER_GRADIENT_DESCENT_H_

#include "optimizer.h"

namespace auto_diff {
namespace optimizer {

class GradientDescent : public Optimizer {
public:
  GradientDescent(){};
  GradientDescent(const Node &target, const size_t &batch_size = 12,
                  const double &learning_rate = 0.01, Graph *graph = nullptr);

private:
  double learning_rate_;

  void update(); // update the gradient to parameters
};
} // namespace optimizer
} // namespace auto_diff

#endif