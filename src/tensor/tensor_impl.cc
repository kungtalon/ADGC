#include "tensor/tensor.h"

namespace tensor {
template<>
Tensor<double> Tensor<double>::operator/(const double &denom) const {
  double devided = 1 / denom;
  if (std::abs(devided) < 1e-17) {
    throw adg_exception::DividingZeroException();
  }
  return this->multiply(devided);
}

template<>
Tensor<float> Tensor<float>::operator/(const float &denom) const {
  float devided = 1 / denom;
  if (std::abs(devided) < 1e-15) {
    throw adg_exception::DividingZeroException();
  }
  return this->multiply(devided);
}

template<>
Tensor<int32_t> Tensor<int32_t>::operator/(const int32_t &denom) const {
  if (denom == 0) {
    throw adg_exception::DividingZeroException();
  }

  Tensor<int32_t> result = this->copy();
  result.map([](int32_t &val) { val /= val; });
  return result;
}
} // namespace tensor
