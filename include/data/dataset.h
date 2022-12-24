//
// Created by kungtalon on 2022/12/23.
//

#ifndef ADGC_DATA_DATASET_H_
#define ADGC_DATA_DATASET_H_

#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <functional>
#include <unordered_set>
#include <cassert>

#include "tensor/tensor.h"
#include "data_utils.h"

namespace auto_diff {
typedef std::pair<tensor::Tensor<double>, tensor::Tensor<double>> DataPair;
}

#include "image_dataset.h"
#include "csv_dataset.h"

#endif //ADGC_DATA_DATASET_H_
