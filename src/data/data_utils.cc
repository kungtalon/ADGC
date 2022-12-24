//
// Created by Kungtalon on 2022/12/22.
//

#include "assert.h"
#include "data/data_utils.h"

namespace auto_diff {
namespace data {

tensor::Tensor<double> to_one_hot(const size_t &label_num, const tensor::Tensor<double> &label_tensor) {
  size_t len = label_tensor.get_shape()[0];

  assert(len == label_tensor.get_size());

  tensor::Tensor<double> result({label_tensor.get_size(), label_num}, 0.);
  for (size_t ix = 0; ix < label_tensor.get_size(); ++ix) {
    size_t label_ind = label_tensor.get_value();
    result.set_value({ix, label_ind}, 1.);
  }
  return result;
}

}
}