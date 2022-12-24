//
// Created by kungtalon on 2022/12/23.
//

#ifndef ADGC_AUTODIFF_METRIC_METRIC_H_
#define ADGC_AUTODIFF_METRIC_METRIC_H_

#include "tensor/tensor.h"

namespace auto_diff {

namespace metric {

template<typename dType>
double accuracy(const tensor::Tensor<dType> &pred_ts, const tensor::Tensor<dType> &labels, bool normalize = true) {
  size_t len = labels.get_shape()[0];
  size_t size = labels.get_size();
  double res = 0;

  if (len == size) {
    // labels is a one-dim tensor
    for (size_t ix = 0; ix < len; ++ix) {
      if (pred_ts.get_value({ix}) == labels.get_value({ix})) {
        res += 1;
      }
    }
  } else {
    // labels is a two-dim one-hot matrix
    auto int_preds = pred_ts.to_int().to_vector();

    for (size_t ix = 0; ix < len; ++ix) {
      if (labels.get_value({ix, (size_t) int_preds[ix]}) == 1.) {
        res += 1;
      }
    }
  }

  if (normalize) {
    res /= len;
  }
  return res;
}
}

}

#endif //ADGC_AUTODIFF_METRIC_METRIC_H_
