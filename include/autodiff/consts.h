#ifndef ADGC_AUTODIFF_CONSTS_H_
#define ADGC_AUTODIFF_CONSTS_H_

#include <string>

#define ADG_DEBUG_GLOABL_BOOL_ false

struct NodeType {
  static inline const std::string ADG_UNKNOWN_TYPE = "unknown";
  static inline const std::string ADG_VARIABLE_TYPE = "variable";
  static inline const std::string ADG_PARAMETER_TYPE = "parameter";

  // ops
  static inline const std::string ADG_MATMUL_TYPE = "OP_matmul";
  static inline const std::string ADG_VECDOT_TYPE = "OP_vecdot";
  static inline const std::string ADG_ADD_TYPE = "OP_add";
  static inline const std::string ADG_MATADDVEC_TYPE = "OP_mataddvec";
  static inline const std::string ADG_MATSUM_TYPE = "OP_matsum";
  static inline const std::string ADG_RESHAPE_TYPE = "OP_reshape";
  static inline const std::string ADG_POINTMUL_TYPE = "OP_pointmul";
  static inline const std::string ADG_PAD2D_TYPE = "OP_pad2d";
  static inline const std::string ADG_CONV2D_TYPE = "OP_conv2d";

  // functional
  static inline const std::string ADG_SIGMOID_TYPE = "F_sigmoid";
  static inline const std::string ADG_RELU_TYPE = "F_relu";
  static inline const std::string ADG_CROSS_ENTROPY_SOFTMAX_TYPE =
    "F_cross_entropy_softmax";
  static inline const std::string ADG_REDUCE_SUM_TYPE = "F_reduce_sum";
  static inline const std::string ADG_REDUCE_MEAN_TYPE = "F_reduce_mean";
};

struct LayerType {
  static inline const std::string ADG_LAYER_TYPE = "L_layer";
  static inline const std::string ADG_LAYER_DENSE = "L_dense";
  static inline const std::string ADG_LAYER_CONV2D = "L_conv2d";
  static inline const std::string ADG_LAYER_RELU = "L_relu";
  static inline const std::string ADG_LAYER_SIGMOID = "L_sigmoid";
  static inline const std::string ADG_LAYER_CROSS_ENTROPY_SOFTMAX = "L_cross_entropy_softmax";
};

const inline std::string VAR_GRAPHVIZ_NODE_COLOR = "deepskyblue1";
const inline std::string OPS_GRAPHVIZ_NODE_COLOR = "darkorange2";
const inline std::string FUNC_GRAPHVIZ_NODE_COLOR = "mediumseagreen";
const inline std::string OTHER_GRAPHVIZ_NODE_COLOR = "plum";

#endif