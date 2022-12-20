#include "autodiff/graph.h"
#include "autodiff/node.h"

namespace graph_component {

Graph::Graph(){};

Graph::Graph(const std::string &name) : graph_name_(name) {}

Graph::~Graph() {}

void Graph::remove_all() {
  // delete all node pointers by graph
  // don't delete nodes elsewhere
  for (auto map_iter = node_ptr_dict_.begin(); map_iter != node_ptr_dict_.end();
       ++map_iter) {
    delete map_iter->second;
  }
  node_ptr_dict_.clear();
  type_counter_.clear();
}

std::string Graph::add_node(Node *node, const std::string &type,
                            const std::string &name) {
  std::string valid_node_name = name;

  if (valid_node_name == "") {
    // use type counter to generate a incremental id as node name
    std::string id_as_name = std::to_string(type_counter_.inc(type));
    valid_node_name = id_as_name;
  }

  std::string full_node_name = type + "_" + valid_node_name;

  if (contains_node(full_node_name)) {
    throw adg_exception::DuplicateNodeNameError(
        "Found duplicated node names: " + full_node_name);
  }

  node_ptr_dict_[full_node_name] = node;
  node_ptr_list_.push_back(node);
  return valid_node_name;
}

bool Graph::contains_node(const std::string &full_node_name) const {
  return node_ptr_dict_.find(full_node_name) != node_ptr_dict_.end();
}

Graph *Graph::get_instanceof_global_graph() {
  if (global_graph == nullptr) {
    global_graph = new Graph();
  }
  return global_graph;
}

void Graph::clear_all_jacobi() {
  for (auto node_ptr : node_ptr_list_) {
    node_ptr->clear_jacobi();
  }
}

void Graph::clear_all_value() {
  for (auto node_ptr : node_ptr_list_) {
    node_ptr->clear_value(false);
  }
}

#ifdef ADGC_ENABLE_GRAPHVIZ_
// use graphviz to plot the graph
void Graph::visualize(const std::string &file_name) {
  std::string g_name = graph_name_.empty() ? "graph" : graph_name_;
  auto gv_tool = GraphVizTool(g_name);
  std::unordered_map<std::string, Agnode_t *> name_to_agnode;

  auto get_agnode_t = [&](const std::string &node_name,
                          const std::string &node_type) mutable {
    if (name_to_agnode.find(node_name) == name_to_agnode.end()) {
      name_to_agnode[node_name] = gv_tool.add_node(node_name, node_type);
    }
    return name_to_agnode[node_name];
  };

  for (auto map_iter = node_ptr_dict_.begin(); map_iter != node_ptr_dict_.end();
       ++map_iter) {
    std::string full_node_name = map_iter->first;
    Agnode_t *cur_agnode_t =
        get_agnode_t(full_node_name, map_iter->second->get_type());
    Node *cur_node = map_iter->second;

    Agnode_t *child_agnode_t;
    for (auto child_ptr : cur_node->get_children()) {
      child_agnode_t =
          get_agnode_t(child_ptr->get_full_name(), child_ptr->get_type());
      gv_tool.add_edge(cur_agnode_t, child_agnode_t);
    }

    if (cur_node->get_children().empty()) {
      gv_tool.set_node_attr(cur_agnode_t, "rank", "max");
    }

    gv_tool.layout();
    // gv_tool.render();
    gv_tool.render_file(file_name.c_str());
  }
}
#endif

} // namespace graph_component