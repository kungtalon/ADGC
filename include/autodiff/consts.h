#ifndef ADGC_AUTODIFF_CONSTS_H_
#define ADGC_AUTODIFF_CONSTS_H_

#include <string>

#define ADG_DEBUG_GLOABL_BOOL_ false

namespace auto_diff {

struct NodeType {
  // base
  static inline const std::string ADG_UNKNOWN_TYPE = "unknown";
  static inline const std::string ADG_VARIABLE_TYPE = "variable";
  static inline const std::string ADG_PARAMETER_TYPE = "parameter";

  // ops
  static inline const std::string ADG_MATMUL_TYPE = "OP_mat_mul";
  static inline const std::string ADG_VECDOT_TYPE = "OP_vec_dot";
  static inline const std::string ADG_ADD_TYPE = "OP_add";
  static inline const std::string ADG_MATADDVEC_TYPE = "OP_mat_add_vec";
  static inline const std::string ADG_MATSUM_TYPE = "OP_mat_sum";
  static inline const std::string ADG_POINTMUL_TYPE = "OP_point_mul";

  // normalization
  static inline const std::string ADG_BATCHNORM2D_TYPE = "OP_batch_norm";

  // vision ops
  static inline const std::string ADG_CONV2D_TYPE = "OP_conv2d";

  // unary op
  static inline const std::string ADG_RESHAPE_TYPE = "OP_reshape";
  static inline const std::string ADG_PAD2D_TYPE = "OP_pad2d";

  // activation
  static inline const std::string ADG_SIGMOID_TYPE = "F_sigmoid";
  static inline const std::string ADG_RELU_TYPE = "F_relu";

  // loss
  static inline const std::string ADG_CROSS_ENTROPY_SOFTMAX_TYPE =
    "F_cross_entropy_softmax";

  // reduction
  static inline const std::string ADG_REDUCE_SUM_TYPE = "F_reduce_sum";
  static inline const std::string ADG_REDUCE_MEAN_TYPE = "F_reduce_mean";
};

struct LayerType {
  static inline const std::string ADG_LAYER_TYPE = "L_layer";
  static inline const std::string ADG_LAYER_DENSE = "L_dense";
  static inline const std::string ADG_LAYER_CONV2D = "L_conv2d";
  static inline const std::string ADG_LAYER_BATCHNORM2D = "L_batchnorm2d";
  static inline const std::string ADG_LAYER_RELU = "L_relu";
  static inline const std::string ADG_LAYER_SIGMOID = "L_sigmoid";
  static inline const std::string ADG_LAYER_CROSS_ENTROPY_SOFTMAX = "L_cross_entropy_softmax";
};

struct GraphViz {
  static const inline std::string VAR_GRAPHVIZ_NODE_COLOR = "deepskyblue1";
  static const inline std::string OPS_GRAPHVIZ_NODE_COLOR = "darkorange2";
  static const inline std::string FUNC_GRAPHVIZ_NODE_COLOR = "mediumseagreen";
  static const inline std::string OTHER_GRAPHVIZ_NODE_COLOR = "plum";
};

enum class GraphStageFlag {
  train = 0,
  eval = 1
};

}
#endif