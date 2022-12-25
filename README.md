# ADGC
This repo implements a tensor module and a deep learning framework. It aims to help everyone grasp a better and clearer understanding of tensor operations and automatic differentiation mechanisms in TensorFlow, PyTorch etc. The three-layer MLP built on top of my framework has achieved 86% accuracy on MNIST dataset.

## Tensor Module
This part implements a standalone module for creating and manipulating multidimentional array like Numpy. The operations are based on CPU with BLAS.

Some of the methods implemented:
- dot, multiply, add: do some basic algorithm functions for matrices/tensors
- reshape: change the shape of the tensor
- transpose: switch two axes of the tensor
- slice: slice tensor along several axes with given indices
- sum: sum the values along an axis
- fill_diag: fill the diagonal entries with a vector
- map: accept a lambda function and transforms the value of each entry
- kron: do the matrix kronecker product

We are trying to learn BLAS and make full use of its extreme performance. Any advice would be very appreciated! :)

## Deep Learning Auto-Differentiation Framework
This part contains all elements with regard to building an automatic differential graph and a deep learning model trainer. There are different types of nodes on the graph:
- Node: base class for all nodes; it is abstract and no backward or forward is defined
- Variable: leaf nodes controlling the user's input
- Parameter: trainable values of the neural networks
- Ops: algorithmic operations between different nodes
- Functional: activation functions and loss functions

These are not nodes but important to building and training the model:
- Layer: some common layers built on top of the above nodes, like fully connected layer
- Optimizer: encapsulations of gradient-based learning algorithms such as Gradient Descent, Adam...

### Forward and Backward
Each node of the graph needs to implement the logics of `do_forward()` and `do_backward(Node* parent_ptr)`, the first one computes its forward value and the other computes the nodes' jacobian matrix with regard to one of its parent nodes.

A code snippet about how to use the framework:

```cpp
  Variable x = Variable({2, 2});       
  Variable labels = Variable({2, 2}); 

  tensor::Tensor<double> value_x = tensor::Tensor<double>({2, 2}, {1, 2, 3, 4});
  tensor::Tensor<double> value_labels = tensor::Tensor<double>({2, 2}, {1, 0, 1, 0});

  // build graph
  layer::Dense dense_layer_1(2, 10, "relu");
  layer::Dense dense_layer_2(10, 2, "none");

  auto loss = functional::cross_entropy_with_softmax(
    dense_layer_2(dense_layer_1(x)), labels);

  auto optim = optimizer::Adam(loss);

  // forward
  x.assign_value(value_x);
  labels.assign_value(value_labels);
  optim.zero_grad();
  loss.forward();

  // backward
  optim.step();

  // in the end
  Graph::clear_graph();
```

### Demo
There are runnable demos in the directory `demo`. You can build and run it yourself with the CMakeLists.txt given. Please make sure the dependencies are installed properly.
```shell
mkdir build
cd build
cmake -DSKIP_TEST=ON ..
make mnist_trainer
./mnist_trainer $BATCH_SIZE $EPOCH $TRAIN_DATA_PATH $TEST_DATA_PATH
```


### Graph Visualization
We are using [Graphviz](https://graphviz.org/about/) to visualize the computational graph. Call `graph->visualize("file.svg")` to convert the graph into a DOT image and saves as SVG file.

Use `-DUSE_GRAPHVIZ=OFF` in cmake command line to turn off using graphviz.

Example output:

<img src="graphviz/test.svg" alt="graphviz_out" style="width:700px;"/>

## Dependency
Here are some software and packages used in this project:

> cblas
>
> graphviz
>
> cgraph
>
> googletest
