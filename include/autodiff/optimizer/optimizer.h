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
            const double &learning_rate = 0.01, Graph *graph = nullptr);

  void zero_grad();
  void step();

protected:
  const size_t batch_size_;
  Graph *graph_;
  Node *target_node_ptr_;
  std::unordered_map<std::string, DTensor>
      acc_grads_; // accumulated mini-batch gradients
  size_t acc_counter_;
  double learning_rate_;

  bool is_trainable_param(Node *node_ptr);
  DTensor get_gradient(Node *node_ptr); // get the mini-batched average gradient
  virtual void update() = 0;            // update the gradient to parameters
  void propagate();                     // do forward propagation and backward
};
} // namespace optimizer
} // namespace auto_diff

#include "adam.h"
#include "gradient_descent.h"

#endif