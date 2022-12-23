//
// Created by Zelon on 2022/12/22.
//

#ifndef ADGC_INCLUDE_DATA_DATA_UTILS_H_
#define ADGC_INCLUDE_DATA_DATA_UTILS_H_

#include "tensor/tensor.h"

namespace auto_diff {

namespace data {

tensor::Tensor<double> to_one_hot(const size_t &label_num, const tensor::Tensor<double> &label_tensor);

}
}

#endif //ADGC_INCLUDE_DATA_DATA_UTILS_H_
