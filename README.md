# Auto-Diff-Graph-CPP
This repo aims to help everyone grasp a deeper understanding of tensor operations and automatic differentiation mechanisms in TensorFlow, PyTorch etc.

## Tensor Module
This part implements a standalone module for creating and manipulating multidimentional array like Numpy. The operations are based on CPU with BLAS.

Some of the methods implemented:
- dot, multiply, add: some basic algorithm functions for matrices/tensors
- reshape: change the shape of the tensor
- transpose: switch two axes of the tensor
- take: slice tensor along an axis
- sum: sum the values along an axis
- fill_diag: fill the diagonal entries with a vector
- map: accepts a lambda function and transforms the value of each entry

We are trying to learn BLAS and use it for as many as we can. Any advice would be very appreciated! :)

## Graph Module
This part contains all elements with regard to building an automatic differential graph. There are different types of nodes on the graph:
- Node: base class for all nodes; it is abstract and no backward or forward is defined
- Variable: leaf nodes controlling the user's input
- Ops: operations between different nodes
- Functional: activation functions and loss functions

### Graph Visualization
We are using [Graphviz](https://graphviz.org/about/) to visualize the computational graph. Call `graph.visualize("file.svg")` to convert the graph into a DOT image and saves as SVG file.

Use `-DUSE_GRAPHVIZ=OFF` in cmake command line to turn off using graphviz.

## Dependencies
Here are some software and packages used in this project:

> cblas
>
> graphviz
>
> cgraph
>
> googletest
