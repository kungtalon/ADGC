#ifndef ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_
#define ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_

#include <string>
#include <deque>

#include "autodiff/component/node.h"
#include "autodiff/component/variable.h"

namespace auto_diff {
namespace optimizer {

class Optimizer {
 public:
  Optimizer() {};
  Optimizer(const Node &target, const double &learning_rate = 0.01, Graph *graph = nullptr);

  void zero_grad();
  void step();
  void set_requires_grads_for_all();

 protected:
  Graph *graph_;
  Node *target_node_ptr_;
  std::unordered_map<std::string, DTensor>
    acc_grads_; // accumulated mini-batch gradients
  double learning_rate_;
  bool get_all_grads_;
  std::vector<Node *> trainable_params_list_;

  void agg_trainable_params();
  DTensor get_gradient(Node *node_ptr); // get the mini-batched average gradient
  virtual void update() = 0;            // update the gradient to parameters
  void propagate();                     // do forward propagation and backward
};
} // namespace optimizer
} // namespace auto_diff

#include "adam.h"
#include "gradient_descent.h"

#endif