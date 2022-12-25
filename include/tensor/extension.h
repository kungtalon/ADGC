//
// Created by kungtalon on 2022/12/24.
//

#ifndef ADGC_INCLUDE_TENSOR_EXTENSION_H_
#define ADGC_INCLUDE_TENSOR_EXTENSION_H_

#include "tensor.h"

namespace tensor {

template<typename dType>
Tensor<dType> pad2d(const Tensor<dType> &src_tensor,
                    const std::vector<size_t> &paddings,
                    const dType &value = 0);

template<typename dType>
Tensor<dType> pad2d(const Tensor<dType> &src_tensor,
                    const std::vector<std::pair<size_t, size_t>> &paddings,
                    const dType &value = 0);

template<typename dType>
static inline Tensor<dType> dot(const Tensor<dType> &lt,
                                const Tensor<dType> &rt) {
  return lt.dot(rt);
}

template<typename dType>
static inline Tensor<dType> multiply(const Tensor<dType> &lt,
                                     const Tensor<dType> &rt) {
  return lt.multiply(rt);
}

template<typename dType>
static inline Tensor<dType> add(const Tensor<dType> &lt,
                                const Tensor<dType> &rt) {
  return lt.add(rt);
}

template<typename dType>
static inline Tensor<dType> sub(const Tensor<dType> &lt,
                                const Tensor<dType> &rt) {
  return lt.sub(rt);
}

template<typename dType>
static inline Tensor<dType> sum(const Tensor<dType> &ts,
                                const size_t &axis = SIZE_MAX) {
  return ts.sum(axis);
}

}
#include "tensor/extension.tcc"

#endif //ADGC_INCLUDE_TENSOR_EXTENSION_H_
