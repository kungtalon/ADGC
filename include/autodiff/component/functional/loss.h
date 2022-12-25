//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_LOSS_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_LOSS_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class CrossEntropyWithSoftMax : public Node {
 public:
  CrossEntropyWithSoftMax() : Node(NodeType::ADG_CROSS_ENTROPY_SOFTMAX_TYPE) {};
  CrossEntropyWithSoftMax(Node *parent_ptr, Variable *labels_ptr,
                          Graph *g = nullptr, const std::string &name = "");
  static DTensor softmax(const DTensor &input);
  DTensor get_probs();
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  static inline double epsilon_ = 1e-9;
  DTensor probs_;
  DTensor neg_log_probs_;
};

CrossEntropyWithSoftMax &
cross_entropy_with_softmax(const Node &input, const Variable &labels,
                           Graph *g = nullptr, const std::string &name = "");

}
}
#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_LOSS_H_
