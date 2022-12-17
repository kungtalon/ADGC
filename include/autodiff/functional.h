#ifndef ADGC_AUTODIFF_FUNCTIONAL_H_
#define ADGC_AUTODIFF_FUNCTIONAL_H_


#include "node.h"

namespace graph_component {
namespace functional {

class Logistic : public Node {
public:
    Logistic(): Node("logistic") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
};

class ReLU : public Node {
public:
    ReLU(): Node("relu") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
};

class CrossEntropyWithSoftMax : public Node {
public:
    CrossEntropyWithSoftMax(): Node("cross_entropy_softmax") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
};

}
}


#endif