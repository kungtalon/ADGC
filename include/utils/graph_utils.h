// #define ADGC_ENABLE_GRAPHVIZ_
#ifdef ADGC_ENABLE_GRAPHVIZ_

#ifndef ADGC_UTILS_GRAPH_UTILS_H_
#define ADGC_UTILS_GRAPH_UTILS_H_

#include <array>
#include <string>

#include "graphviz/cgraph.h" // these 2 includes are the graphiz cgraph lib
#include "graphviz/gvc.h"


/*
Credit:
https://codereview.stackexchange.com/questions/236073/c-wrapper-for-graphviz-library
*/
class GraphVizTool {
public:
  GraphVizTool() {
    gvc_ = gvContext();

    static const char *fargv[] = {"dot", "-Tsvg"}; // NOLINT
    gvParseArgs(gvc_, 2, (char **)fargv);          // NOLINT

    graph_ = agopen((char *)"g", Agstrictdirected, nullptr); // NOLINT

    // clang-format off
    // set_graph_attr_def("size",   "20, 20");
    // set_graph_attr_def("splines",   "none");
    // set_graph_attr_def("ratio",     "1.25");

    // set_node_attr_def("tooltip",    "");
    // set_node_attr_def("fillcolor",  "grey");
    // set_node_attr_def("shape",      "point");
    // set_node_attr_def("width",      "0.05");
    // set_node_attr_def("penwidth",   "0");

    // set_edge_attr_def("weight",     "1");
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

  inline void layout() { gvLayoutJobs(gvc_, graph_); }

  inline void render() { gvRenderJobs(gvc_, graph_); }

  inline void render_file(const char *file_name) {
    gvRenderFilename(gvc_, graph_, "svg", file_name);
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