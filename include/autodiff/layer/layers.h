#ifndef ADGC_AUTODIFF_LAYER_LAYERS_H_
#define ADGC_AUTODIFF_LAYER_LAYERS_H_

#include "autodiff/consts.h"
#include "autodiff/functional.h"
#include "autodiff/graph.h"
#include "autodiff/ops.h"
#include "autodiff/variable.h"

namespace graph_component {
namespace layer {

class Layer {
public:
  Layer(){};
  Layer(const std::string &layer_type, Graph *g = nullptr);
  ~Layer(){};

  void freeze();
  std::vector<Parameter *> get_param_ptr_list() const;

protected:
  std::string layer_name_;
  Graph *graph_;
  std::vector<Parameter *> params_ptr_list_;

  void add_param(Parameter *param_ptr);
};

} // namespace layer
} // namespace graph_component

#include "autodiff/layer/dense.h"

#endif