// #define ADGC_ENABLE_GRAPHVIZ_
#ifdef ADGC_ENABLE_GRAPHVIZ_

#ifndef ADGC_UTILS_GRAPH_UTILS_H_
#define ADGC_UTILS_GRAPH_UTILS_H_

#include <iostream>
#include <array>
#include <string>
#include <cstdlib>

#include "graphviz/cgraph.h" // these 2 includes are the graphiz cgraph lib
#include "graphviz/gvc.h"

#include "autodiff/consts.h"

/*
Credit:
https://codereview.stackexchange.com/questions/236073/c-wrapper-for-graphviz-library
*/
class GraphVizTool {
public:
  GraphVizTool(){};

  GraphVizTool(const std::string& graph_name = "g") {
    gvc_ = gvContext();

    static const char *fargv[] = {"dot", "-Tsvg"}; // NOLINT
    gvParseArgs(gvc_, 2, (char **)fargv);          // NOLINT

    std::vector<char> cstr(graph_name.c_str(), graph_name.c_str() + graph_name.size() + 1);
    graph_ = agopen(&*cstr.begin(), Agstrictdirected, nullptr); // NOLINT

    // clang-format off
    // set_graph_attr_def("size",   "20, 20");
    // set_graph_attr_def("splines",   "none");
    // set_graph_attr_def("ratio",     "1.25");
    set_graph_attr_def("rankdir", "LR");
    set_graph_attr_def("nodesep", "0.4");

    // set_node_attr_def("tooltip",    "");
    set_node_attr_def("fillcolor",  "plum");
    set_node_attr_def("shape",      "circle");
    set_node_attr_def("width",      "0.02");
    set_node_attr_def("penwidth",   "1.4");
    set_node_attr_def("style", "filled");
    set_node_attr_def("fontsize", "20");
    set_node_attr_def("color", "none");

    set_edge_attr_def("weight",     "0.05");
    set_edge_attr_def("color",     "gray");
    // clang-format on
  }

  GraphVizTool(const GraphVizTool &other) = delete;
  GraphVizTool &operator=(const GraphVizTool &other) = delete;

  GraphVizTool(GraphVizTool &&other) = delete;
  GraphVizTool &operator=(GraphVizTool &&other) = delete;

  ~GraphVizTool() {
    if (graph_ != nullptr) {
      if (gvc_ != nullptr)
        gvFreeLayout(gvc_, graph_);
      agclose(graph_);
    }
    if (gvc_ != nullptr)
      gvFreeContext(gvc_);
  }

  inline void set_graph_attr_def(std::string_view name,
                                 std::string_view value) {
    agattr(graph_, AGRAPH, (char *)name.data(), (char *)value.data()); // NOLINT
  }

  inline void set_node_attr_def(std::string_view name, std::string_view value) {
    agattr(graph_, AGNODE, (char *)name.data(), (char *)value.data()); // NOLINT
  }

  inline void set_edge_attr_def(std::string_view name, std::string_view value) {
    agattr(graph_, AGEDGE, (char *)name.data(), (char *)value.data()); // NOLINT
  }

  inline void set_node_attr(Agnode_t *node, std::string_view name,
                            std::string_view value) {       // NOLINT
    agset(node, (char *)name.data(), (char *)value.data()); // NOLINT
  }

  inline void set_edge_attr(Agedge_t *edge, std::string_view name,
                            std::string_view value) {       // NOLINT
    agset(edge, (char *)name.data(), (char *)value.data()); // NOLIN
  }

  inline Agedge_t *add_edge(Agnode_t *src, Agnode_t *dest,
                            std::string_view weight_str) {
    auto edge = agedge(graph_, src, dest, nullptr, 1);
    set_edge_attr(edge, "weight", weight_str);
    return edge;
  }

  inline Agedge_t *add_edge(Agnode_t *src, Agnode_t *dest) {
    auto edge = agedge(graph_, src, dest, nullptr, 1);
    return edge;
  }

  inline Agnode_t *add_node(std::string_view node_name) {
    auto node = agnode(graph_, (char *)node_name.data(), 1); // NOLINT
    set_node_attr(node, "label", node_name);
    return node;
  }

  inline Agnode_t *add_node(std::string_view node_name, const std::string& node_type) {
    auto node = agnode(graph_, (char *)node_name.data(), 1); // NOLINT

    if (node_type == "variable") {
      set_node_attr(node, "label", node_name);
      set_node_attr(node, "fillcolor", VAR_GRAPHVIZ_NODE_COLOR);
      set_node_attr(node, "rank", "min");
    } else if (node_type.rfind("OP") == 0) {
      // type starts with op, ops
      set_node_attr(node, "label", node_name.substr(3, node_name.size() - 3));
      set_node_attr(node, "fillcolor", OPS_GRAPHVIZ_NODE_COLOR);
    } else if (node_type.rfind("F") == 0) {
      // type starts with F, functional
      set_node_attr(node, "label", node_name.substr(2, node_name.size() - 2));
      set_node_attr(node, "fillcolor", FUNC_GRAPHVIZ_NODE_COLOR);
    } else {
      set_node_attr(node, "fillcolor", OTHER_GRAPHVIZ_NODE_COLOR);
    }
    return node;
  }

  inline void layout() { gvLayoutJobs(gvc_, graph_); }

  inline void render() { gvRenderJobs(gvc_, graph_); }

  inline void render_file(const std::string& file_name) {
    int name_len = file_name.size();
    if (name_len <= 3) {
      std::cerr << "Invalid file name for graphviz" << std::endl;
    }
    if (file_name[name_len - 1] == 'g' && file_name[name_len -2] == 'v') {
      gvRenderFilename(gvc_, graph_, "dot", "graphviz.dot");
      std::string run_command = "cat graphviz.dot | dot -Tsvg -o " + file_name;
      system(run_command.c_str());
    } else if (file_name[name_len - 1] == 't' && file_name[name_len -2] == 'o') {
      gvRenderFilename(gvc_, graph_, "dot", file_name.c_str());
    } else {
      std::cerr << "Invalid file name for graphviz" << std::endl;
    }
  }

private:
  Agraph_t *graph_ = nullptr;
  GVC_t *gvc_ = nullptr;
};

static constexpr const size_t max_colours = 30;

static constexpr const std::array<std::string_view, max_colours> colours = {
    "blue",           "green",         "red",        "gold",
    "black",          "magenta",       "brown",      "pink",
    "khaki",          "cyan",          "tan",        "blueviolet",
    "burlywood",      "cadetblue",     "chartreuse", "chocolate",
    "coral",          "darkgoldenrod", "darkgreen",  "darkkhaki",
    "darkolivegreen", "darkorange",    "darkorchid", "darksalmon",
    "darkseagreen",   "dodgerblue",    "lavender",   "mediumpurple",
    "plum",           "yellow"};

#endif

#endif