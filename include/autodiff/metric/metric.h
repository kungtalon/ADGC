//
// Created by kungtalon on 2022/12/23.
//

#ifndef ADGC_AUTODIFF_METRIC_METRIC_H_
#define ADGC_AUTODIFF_METRIC_METRIC_H_

#include "tensor/tensor.h"

template<typename dType>
double Accuracy(const tensor::Tensor<dType> &pred_ts, const tensor::Tensor<dType> &labels) {
  size_t len = labels.shape()[0];
  size_t size = labels.size();
  double res = 0;

  if (len == size) {
    // labels is a one-dim tensor
    auto int_labels = labels.to_int().to_vector();

    for (int ix = 0; ix < len; ++ix) {
      if (pred_ts.get_value({ix, int_labels[ix]}) == 1.) {
        res += 1;
      }
    }
  } else {
    // labels is a two-dim one-hot matrix
    res += pred_ts.multiply(labels).sum();
  }

  return res / len;
}

#endif //ADGC_AUTODIFF_METRIC_METRIC_H_
