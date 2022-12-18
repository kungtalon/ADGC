#include <string>

struct NodeType {
  static inline const std::string ADG_UNKNOWN_TYPE = "unknown";
  static inline const std::string ADG_VARIABLE_TYPE = "variable";

  // ops
  static inline const std::string ADG_MATMUL_TYPE = "matmul";
  static inline const std::string ADG_VECDOT_TYPE = "vecdot";
  static inline const std::string ADG_ADD_TYPE = "add";

  // functional
  static inline const std::string ADG_SIGMOID_TYPE = "sigmoid";
  static inline const std::string ADG_RELU_TYPE = "relu";
  static inline const std::string ADG_CROSS_ENTROPY_SOFTMAX_TYPE =
      "cross_entropy_softmax";
};
