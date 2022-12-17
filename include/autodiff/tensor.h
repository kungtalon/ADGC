#ifndef ADGC_AUTODIFF_TENSOR_H_
#define ADGC_AUTODIFF_TENSOR_H_

#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "exception/exception.h"
#include "utils/math_utils.h"
#include "utils/thread.h"
#include "utils/utils.h"

namespace tensor {

typedef std::vector<size_t> TensorShape;
typedef std::vector<size_t> TensorIndex;
template <typename dType> using TensorIterator = std::vector<dType>::iterator;

const TensorShape EMPTY_SHAPE = {0};

#if ENABLE_TENSOR_MULTI_THREAD
static const size_t MAX_THREAD_NUM = 4;
#endif

template <typename dType> class Tensor {
public:
  // Tensor();
  Tensor(const TensorShape &shape);
  Tensor(const TensorShape &shape, const dType &single_value);
  Tensor(const TensorShape &shape, dType *values);
  Tensor(const TensorShape &shape, const std::vector<dType> &values);
  Tensor(const Tensor<dType> &another);
  Tensor(const Tensor<dType> &&another);

  Tensor<dType> &operator=(const Tensor<dType> &bt);

  void set_value(const TensorIndex &index, const dType &value);
  dType get_value(const TensorIndex &index);
  void reshape(const TensorShape &new_shape);
  Tensor<dType> dot(const Tensor<dType> &bt);
  Tensor<dType> multiply(const double &multiplier);
  Tensor<dType> multiply(const Tensor<dType> &bt);
  Tensor<dType> add(const Tensor<dType> &bt);
  Tensor<dType> add(const double &number);
  Tensor<dType> transpose();
  Tensor<dType> transpose(const size_t &axis_a, const size_t &axis_b);
  Tensor<dType> copy();
  void normal_init(double loc = 0., double scale = 1., size_t seed = SIZE_MAX);

#if TENSOR_TESTING

  inline std::vector<dType> test_get_tensor() {
    return *static_cast<std::vector<dType> *>(tensor_.get());
  };

#endif

  inline TensorShape get_shape() const { return shape_; };
  inline size_t get_size() const { return size_; };
  inline size_t get_dim() const { return dim_; };
  inline TensorShape get_strides() const { return strides_; };
  inline std::string to_string() const { return do_to_string(); };

protected:
  // store tensor as a vector, wrapped in shared_ptr for easy copy
  std::shared_ptr<std::vector<dType>> tensor_;
  TensorShape shape_;
  TensorShape strides_;
  size_t size_;
  size_t dim_;

  bool is_shape_valid(const TensorShape &shape) const;
  bool is_index_valid(const TensorIndex &index) const;

  // helper functions
  TensorIterator<dType> get_iterator(const TensorIndex &index);
  TensorShape get_dot_shape(const Tensor<dType> &bt) const;
  static size_t get_coordinate_at_axis(const size_t &ind, const size_t &axis,
                                       const TensorShape &strides);
  static size_t get_index_after_transpose(const size_t arr_ind,
                                          const size_t &axis_a,
                                          const size_t &axis_b,
                                          const TensorShape &ori_strides,
                                          const TensorShape &new_strides);
  inline dType *get_tensor_ptr() { return &*tensor_->begin(); };
  inline const dType *get_tensor_const_ptr() const {
    return &*tensor_->begin();
  };

  // impl functions
  void do_transpose(const size_t &axis_a, const size_t &axis_b,
                    Tensor<dType> &dest_tensor);
  inline std::string do_to_string() const {
    dType *raw_tensor_ptr = &*tensor_->begin();
    return utils::multi_array_to_str(shape_, raw_tensor_ptr);
  }
};

class Zeros : public Tensor<double> {
  Zeros(const TensorShape &shape) : Tensor<double>(shape, 0.){};
};

class Ones : public Tensor<double> {
  Ones(const TensorShape &shape) : Tensor<double>(shape, 1.){};
};

class Eye : public Tensor<double> {
  Eye(const size_t &len) : Tensor<double>({len, len}, 0.) {
    for (size_t ix = 0; ix < len; ix++) {
      set_value({ix, ix}, 1.);
    }
  }
};

} // namespace tensor

#include "autodiff/tensor.tcc"

#endif
