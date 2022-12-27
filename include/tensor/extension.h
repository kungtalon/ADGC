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
Tensor<dType> dilate2d(const Tensor<dType> &src_tensor,
                       const std::array<size_t, 2> &gaps,
                       const dType &value = 0);

template<typename dType>
void reverse(Tensor<dType> &ts, const size_t &axis);

template<typename dType>
dType squared_sum(const Tensor<dType> &ts);

template<typename dType>
Tensor<dType> sqrt(const Tensor<dType> &ts);

template<typename dType>
Tensor<dType> add_vec(const Tensor<dType> &lt,
                      const Tensor<dType> &rt,
                      const size_t &axis);

template<typename dType>
Tensor<dType> dot(const Tensor<dType> &lt,
                  const Tensor<dType> &rt) {
  return lt.dot(rt);
}

template<typename dType>
Tensor<dType> multiply(const Tensor<dType> &lt,
                       const Tensor<dType> &rt) {
  return lt.multiply(rt);
}

template<typename dType>
Tensor<dType> div(const Tensor<dType> &lt,
                  const Tensor<dType> &rt);

template<typename dType>
Tensor<dType> add(const Tensor<dType> &lt,
                  const Tensor<dType> &rt) {
  return lt.add(rt);
}

template<typename dType>
Tensor<dType> sub(const Tensor<dType> &lt,
                  const Tensor<dType> &rt) {
  return lt.sub(rt);
}

template<typename dType>
Tensor<dType> sum(const Tensor<dType> &ts,
                  const size_t &axis = SIZE_MAX) {
  return ts.sum(axis);
}

}
#include "tensor/extension.tcc"

#endif //ADGC_INCLUDE_TENSOR_EXTENSION_H_
