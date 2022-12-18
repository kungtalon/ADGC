#include "autodiff/functional.h"

namespace graph_component {

namespace functional {

Logistic::Logistic(const std::vector<Node *> &parents) : Node("add", parents) {
  if (parents.size() != 1) {
    throw adg_exception::FunctionalParentsNumException("Logistic ==> Logistic");
  }
}

void Logistic::do_forward() {
  if (parents_.empty()) {
    throw adg_exception::FunctionalParentsUnsetException(
        "Logistic ==> do_forward");
  }
  DTensor x = parents_[0]->get_value();
  value_ = (1 / 1 + power(e, np.where(-x > 1e2, 1e2, -x)))
};

} // namespace functional

} // namespace graph_component