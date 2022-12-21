#ifndef ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_
#define ADGC_AUTODIFF_OPTIMIZER_OPTIMIZER_H_

#include "autodiff/component/node.h"


namespace auto_diff {
    namespace optimizer {

        class Optimizer {
            public:
    Optimizer(){};
    Optimizer(const Node& target, const size_t& batch_size=12, Graph* graph = nullptr);
    Optimizer(const Optimizer& other) = delete;
    Optimizer(const Optimizer&& other) = delete;

    void step();
    DTensor get_gradient();
    virtual void update() = 0;
    void forward_backward();

protected:
    size_t batch_size_;
    Graph* graph_;
    Node* target_node_ptr_;
    std::unordered_map<Node*, double> acc_grads_; // accumulated mini-batch gradients
    size_t acc_counter_;  

};
}
}

#endif