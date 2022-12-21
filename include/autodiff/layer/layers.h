#ifndef ADGC_AUTODIFF_LAYER_LAYERS_H_
#define ADGC_AUTODIFF_LAYER_LAYERS_H_

#include "autodiff/component/functional.h"
#include "autodiff/component/ops.h"
#include "autodiff/component/variable.h"
#include "autodiff/consts.h"
#include "autodiff/graph.h"

namespace auto_diff {
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
} // namespace auto_diff

#include "autodiff/layer/dense.h"

#endif