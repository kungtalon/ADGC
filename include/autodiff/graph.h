#ifndef ADGC_AUTODIFF_GRAPH_H_
#define ADGC_AUTODIFF_GRAPH_H_

#include <stdlib.h>

#include <vector>

#include "node.h"

namespace graph_component {

class Graph {
private:
public:
  static Graph *global_graph;

  Graph();
  ~Graph();
  void add_node(Node *n);

  static Graph *get_instanceof_global_graph();
};
} // namespace graph_component
#endif