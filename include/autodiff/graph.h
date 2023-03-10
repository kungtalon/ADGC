#ifndef ADGC_AUTODIFF_GRAPH_H_
#define ADGC_AUTODIFF_GRAPH_H_

#include <stdlib.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/utils.h"

#ifdef ADGC_ENABLE_GRAPHVIZ_
#include "utils/graph_utils.h"
#endif

namespace auto_diff {

class Node;

typedef std::vector<Node *>::iterator NodeIterator;
typedef std::pair<NodeIterator, NodeIterator> NodeIteratorPair;

class Graph {
 public:
  Graph();
  Graph(const std::string &name);
  Graph(const Graph &other) = delete;
  Graph(const Graph &&other) = delete;
  Graph &operator=(const Graph &other) = delete;
  Graph &operator=(const Graph &&other) = delete;

  ~Graph();
  std::string add_node(Node *node, const std::string &type,
                       const std::string &name);
  void add_relation(const std::string &parent_name,
                    const std::string &child_name);
  Node *get_ptr_of(const std::string &node_name);
  bool contains_node(const std::string &full_node_name) const;
  void backward(Node &result);
  void zero_grad();
  void clear_all_value();
  void remove_all();

  inline std::vector<Node *> get_node_list() const { return node_ptr_list_; };
  inline NodeIteratorPair get_node_iterators() {
    return {node_ptr_list_.begin(), node_ptr_list_.end()};
  }
  inline void set_graph_name(const std::string &name) { graph_name_ = name; };
  inline size_t counter_increment(const std::string &key) {
    return type_counter_.inc(key);
  };
  inline void train() { stage_flag_ = GraphStageFlag::train; }
  inline void eval() { stage_flag_ = GraphStageFlag::eval; };
  inline GraphStageFlag stage() { return stage_flag_; };

  static Graph *get_instanceof_global_graph();
  static void clear_graph(Graph *graph = nullptr);
  static Node *get_ptr_of(const std::string &node_name, Graph *graph_ptr);
  static inline void delete_global_graph() {
    if (global_graph != nullptr) {
      delete global_graph;
      global_graph = nullptr;
    }
  }
  static inline Graph *global_graph = nullptr;

#ifdef ADGC_ENABLE_GRAPHVIZ_
  void visualize(const std::string &file_name);
#endif

 private:
  std::string graph_name_;
  std::vector<Node *> node_ptr_list_;
  std::unordered_map<std::string, Node *> node_ptr_dict_;
  utils::TypeCounter type_counter_;
  GraphStageFlag stage_flag_;

};

} // namespace auto_diff
#endif