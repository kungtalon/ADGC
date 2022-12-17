#ifndef ADGC_AUTODIFF_GRAPH_H_
#define ADGC_AUTODIFF_GRAPH_H_

#include <stdlib.h>
#include <unordered_map>
#include <vector>

#include "utils/utils.h"

#ifdef ADGC_ENABLE_GRAPHVIZ_
#include "utils/graph_utils.h"
#endif

namespace graph_component {

class Node;

class Graph {
public:
  Graph();
  Graph(const Graph &other) = delete;
  Graph(const Graph &&other) = delete;
  Graph &operator=(const Graph &other) = delete;
  Graph &operator=(const Graph &&other) = delete;

  ~Graph();
  std::string add_node(Node *node, const std::string &type,
                       const std::string &name);
  bool contains_node(const std::string &full_node_name) const;
  void clear_all_jacobi();
  void clear_all_value();
  static Graph *get_instanceof_global_graph();
  static inline void clear_global_graph() { global_graph->remove_all(); };

  static inline Graph *global_graph = NULL;

#ifdef ADGC_ENABLE_GRAPHVIZ_
  void visualize(const std::string &file_name);
#endif

private:
  std::unordered_map<std::string, Node *> node_ptr_dict_;
  utils::TypeCounter type_counter_;

  void remove_all();
};

} // namespace graph_component
#endif