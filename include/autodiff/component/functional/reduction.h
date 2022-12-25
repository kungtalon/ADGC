//
// Created by kungtalon on 2022/12/25.
//

#ifndef ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_REDUCTION_H_
#define ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_REDUCTION_H_

#include "autodiff/component/functional.h"

namespace auto_diff {
namespace functional {

class ReduceSum : public Node {
 public:
  ReduceSum() : Node(NodeType::ADG_REDUCE_SUM_TYPE) {};
  ReduceSum(Node *parent_ptr, Graph *g = nullptr, const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;
};

class ReduceMean : public Node {
 public:
  ReduceMean() : Node(NodeType::ADG_REDUCE_SUM_TYPE) {};
  ReduceMean(Node *parent_ptr, Graph *g = nullptr,
             const std::string &name = "");
  void do_forward() override;
  DTensor do_backward(Node *parent_ptr) override;

 private:
  double multiplier_;
};

ReduceSum &reduce_sum(const Node &input, Graph *g = nullptr,
                      const std::string &name = "");

ReduceMean &reduce_mean(const Node &input, Graph *g = nullptr,
                        const std::string &name = "");

}
}

#endif //ADGC_INCLUDE_AUTODIFF_COMPONENT_FUNCTIONAL_REDUCTION_H_
