#ifndef ADGC_AUTODIFF_LAYER_LAYERS_H_
#define ADGC_AUTODIFF_LAYER_LAYERS_H_

#include "autodiff/component/functional.h"
#include "autodiff/component/variable.h"
#include "autodiff/consts.h"
#include "autodiff/graph.h"

namespace auto_diff {
namespace layer {

class Layer {
 public:
  Layer() {};
  Layer(const std::string &layer_type, Graph *g = nullptr);
  ~Layer() {};

  void freeze();
  std::vector<Parameter *> get_param_ptr_list() const;
  void set_config(const std::string &key, const std::string &value);
  std::string get_config(const std::string &key);

 protected:
  std::string layer_name_;
  Graph *graph_;
  std::unordered_map<std::string, std::string> configs_;
  std::vector<Parameter *> params_ptr_list_;

  void add_param(Parameter *param_ptr);
  virtual Node *use_activation(Node *input);
};

} // namespace layer
} // namespace auto_diff

#include "dense.h"
#include "convolution.h"
#include "normalization.h"

#endif