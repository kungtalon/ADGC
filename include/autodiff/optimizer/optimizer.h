#ifndef ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_
#define ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_

#include <string>

#include "autodiff/component/node.h"
#include "autodiff/component/variable.h"

namespace auto_diff {
namespace optimizer {

class Optimizer {
public:
  Optimizer() : batch_size_(1){};
  Optimizer(const Node &target, const size_t &batch_size = 12,
            Graph *graph = nullptr);

  void step();

protected:
  const size_t batch_size_;
  Graph *graph_;
  Node *target_node_ptr_;
  std::unordered_map<std::string, DTensor>
      acc_grads_; // accumulated mini-batch gradients
  size_t acc_counter_;

  DTensor get_gradient(Node *node_ptr); // get the mini-batched average gradient
  virtual void update() = 0;            // update the gradient to parameters
  void propagate();                     // do forward propagation and backward
};
} // namespace optimizer
} // namespace auto_diff

#include "gradient_descent.h"

#endif