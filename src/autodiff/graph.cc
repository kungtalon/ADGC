#include "autodiff/graph.h"

namespace graph_component {
Graph::Graph() {}

Graph::~Graph() {}

Graph *Graph::get_instanceof_global_graph() {
  if (global_graph == NULL) {
    global_graph = new Graph();
  }
  return global_graph;
}

} // namespace graph_component