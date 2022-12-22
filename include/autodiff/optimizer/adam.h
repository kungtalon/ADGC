#ifndef ADGC_AUTODIFF_OPTIMIZER_ADAM_H_
#define ADGC_AUTODIFF_OPTIMIZER_ADAM_H_

#include "optimizer.h"

namespace auto_diff {
namespace optimizer {

class Adam : public Optimizer {
public:
  Adam(){};
  Adam(const Node &target, const size_t &batch_size = 12,
       const double &learning_rate = 0.01, const double &beta1 = 0.9,
       const double &beta2 = 0.999, const double &weight_decay = 0,
       const double &epsilon = 1e-8, Graph *graph = nullptr);

  void reset_state();

private:
  double beta1_, beta2_, beta1_power_, beta2_power_, weight_decay_, epsilon_;
  std::unordered_map<std::string, DTensor> moments_table_;

  void update(); // update the gradient to parameters
  DTensor update_moments(const std::string &node_name, const DTensor &grad);
};
} // namespace optimizer
} // namespace auto_diff

#endif