#ifndef ADGC_AUTODIFF_OPS_H_
#define ADGC_AUTODIFF_OPS_H_

#include "node.h"

namespace graph_component {
namespace ops {

class Add : public Node {
public:
    Add(): Node("add") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
};


class VecDot : public Node {
    VecDot(): Node("vecdot") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
} ;

class MatMul : public Node {
    MatMul(): Node("matmul") {};
    void compute() override;
    DTensor get_jacobi(Node *parent) override;
} ;

}

}

#endif