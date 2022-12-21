#include "autodiff/component/ops.h"

namespace auto_diff {
namespace ops {

Add &add(const Node &parent1, const Node &parent2, Graph *g,
         const std::string &name) {
  Add *node_ptr =
      new Add(Graph::get_ptr_of(parent1.get_full_name(), g),
              Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

VecDot &vecdot(const Node &parent1, const Node &parent2, Graph *g,
               const std::string &name) {
  VecDot *node_ptr =
      new VecDot(Graph::get_ptr_of(parent1.get_full_name(), g),
                 Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

MatMul &matmul(const Node &parent1, const Node &parent2, Graph *g,
               const std::string &name) {
  MatMul *node_ptr =
      new MatMul(Graph::get_ptr_of(parent1.get_full_name(), g),
                 Graph::get_ptr_of(parent2.get_full_name(), g), g, name);
  return *node_ptr;
}

MatSum &matsum(const std::vector<Node *> &parents_ptr, Graph *g,
               const std::string &name) {
  MatSum *node_ptr = new MatSum(parents_ptr, g, name);
  return *node_ptr;
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, Graph *g,
               const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g)},
                g, name);
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               Graph *g, const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g),
                 Graph::get_ptr_of(parent_3.get_full_name(), g)},
                g, name);
}

MatSum &matsum(const Node &parent_1, const Node &parent_2, const Node &parent_3,
               const Node &parent_4, Graph *g, const std::string &name) {
  return matsum({Graph::get_ptr_of(parent_1.get_full_name(), g),
                 Graph::get_ptr_of(parent_2.get_full_name(), g),
                 Graph::get_ptr_of(parent_3.get_full_name(), g),
                 Graph::get_ptr_of(parent_4.get_full_name(), g)},
                g, name);
}

} // namespace ops
} // namespace auto_diff