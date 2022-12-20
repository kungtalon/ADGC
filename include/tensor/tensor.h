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
#include "mapper.h"
#include "utils/math_utils.h"
#include "utils/thread.h"
#include "utils/utils.h"

namespace tensor {

typedef std::vector<size_t> TensorShape;
typedef std::vector<size_t> TensorIndex;
typedef std::vector<size_t> TensorSlice;
template <typename dType> using TensorIterator = std::vector<dType>::iterator;

const TensorShape EMPTY_SHAPE = {0};

#if ADGC_MULTI_THREADS_NUM_
static const size_t MAX_THREAD_NUM = 4;
#endif

template <typename dType> class Tensor {
public:
  Tensor();
  Tensor(const TensorShape &shape);
  Tensor(const TensorShape &&shape);
  Tensor(const TensorShape &shape, const dType &single_value);
  Tensor(const TensorShape &shape, const dType *values);
  Tensor(const TensorShape &shape, const std::vector<dType> &values);
  Tensor(const TensorShape &shape, const std::vector<dType> &&values);
  Tensor(const Tensor<dType> &another);
  Tensor(const Tensor<dType> &&another);

  Tensor<dType> &operator=(const Tensor<dType> &bt);
  bool operator==(const Tensor<dType> &bt);
  bool operator!=(const Tensor<dType> &bt);
  Tensor<dType> operator[](const size_t &id);
  Tensor<dType> operator[](const TensorSlice &slice);

  Tensor<dType> take(const size_t &axis, const TensorSlice &slice);
  void set_value(const TensorIndex &index, const dType &value);
  dType get_value() const;
  dType get_value(const TensorIndex &index);
  void reshape(const TensorShape &new_shape);
  void fill_diag(const std::vector<dType> &diag_values);
  void map(Mapper<dType> &mapper);
  void map(Mapper<dType> &&mapper);
  void map(const std::function<void(dType &)> &func);

  Tensor<dType> dot(const Tensor<dType> &bt) const;
  Tensor<dType> multiply(const double &multiplier) const;
  Tensor<dType> multiply(const Tensor<dType> &bt) const;
  Tensor<dType> add(const Tensor<dType> &bt) const;
  Tensor<dType> add(const double &number) const;
  Tensor<dType> sub(const Tensor<dType> &bt) const;
  Tensor<dType> t() const;
  Tensor<dType> transpose() const;
  Tensor<dType> transpose(const size_t &axis_a, const size_t &axis_b) const;
  Tensor<dType> copy() const;
  Tensor<dType> sum(const size_t &axis = SIZE_MAX, bool keep_dim = false) const;
  void normal_init(double loc = 0., double scale = 1., size_t seed = SIZE_MAX);

  TensorShape get_dot_shape(const Tensor<dType> &bt) const;

  Tensor<int32_t> to_int() const;
  Tensor<float> to_float() const;
  Tensor<double> to_double() const;

  inline TensorShape get_shape() const { return shape_; };
  inline size_t get_size() const { return size_; };
  inline size_t get_dim() const { return dim_; };
  inline TensorShape get_strides() const { return strides_; };
  inline std::string to_string() const { return do_to_string(); };
  inline std::vector<dType> to_vector() const {
    return *static_cast<std::vector<dType> *>(tensor_.get());
  };

  static Tensor<dType> kron(const Tensor<dType> &lt, const Tensor<dType> &rt);

  static inline Tensor<dType> dot(const Tensor<dType> &lt,
                                  const Tensor<dType> &rt) {
    return lt.dot(rt);
  }
  static inline Tensor<dType> multiply(const Tensor<dType> &lt,
                                       const Tensor<dType> &rt) {
    return lt.multiply(rt);
  }
  static inline Tensor<dType> add(const Tensor<dType> &lt,
                                  const Tensor<dType> &rt) {
    return lt.add(rt);
  }
  static inline Tensor<dType> sub(const Tensor<dType> &lt,
                                  const Tensor<dType> &rt) {
    return lt.sub(rt);
  }
  static inline Tensor<dType> sum(const Tensor<dType> &ts,
                                  const size_t &axis = SIZE_MAX) {
    return ts.sum(axis);
  }

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
  static size_t get_coordinate_at_axis(const size_t &ind, const size_t &axis,
                                       const TensorShape &strides);
  static size_t get_index_after_transpose(const size_t &arr_ind,
                                          const size_t &axis_a,
                                          const size_t &axis_b,
                                          const TensorShape &ori_strides,
                                          const TensorShape &new_strides);
  inline dType *get_tensor_ptr() { return &*tensor_->begin(); };
  inline const dType *get_tensor_const_ptr() const {
    return &*tensor_->begin();
  };

  // impl functions
  void do_shape_update(const TensorShape &shape, const size_t &keep_size = 0);
  void do_transpose(const size_t &axis_a, const size_t &axis_b,
                    Tensor<dType> &dest_tensor) const;
  inline std::string do_to_string() const {
    dType *raw_tensor_ptr = &*tensor_->begin();
    return utils::multi_array_to_str(shape_, raw_tensor_ptr);
  }
};

class Zeros : public Tensor<double> {
public:
  Zeros(const TensorShape &shape) : Tensor<double>(shape, 0.){};
  Zeros(size_t &) = delete;
};

class Ones : public Tensor<double> {
public:
  Ones(const TensorShape &shape) : Tensor<double>(shape, 1.){};
};

class Eye : public Tensor<double> {
public:
  Eye(const size_t &len) : Tensor<double>({len, len}) {
    for (size_t ix = 0; ix < len; ix++) {
      set_value({ix, ix}, 1.);
    }
  }
};

template <typename dType> class Diagonal : public Tensor<dType> {
public:
  Diagonal(){};
  Diagonal(const std::vector<dType> &values)
      : Tensor<dType>({values.size(), values.size()}) {
    Tensor<dType>::fill_diag(values);
  }
};

template <typename dType> class Ranges : public Tensor<dType> {
public:
  Ranges(){};
  Ranges(const TensorShape &shape, const dType &init)
      : Tensor<dType>(std::move(shape)) {
    size_t size = Tensor<dType>::get_size();
    dType *val = new dType[size];
    std::iota(val, val + size, init);
    memcpy(Tensor<dType>::get_tensor_ptr(), val, sizeof(dType) * size);
  }
};

static const Tensor<int32_t> EMPTY_INT = {{1}};
static const Tensor<float> EMPTY_FLOAT = {{1}};
static const Tensor<double> EMPTY_DOUBLE = {{1}};
static const Tensor<double> EMPTY = {{1}};

} // namespace tensor

#include "tensor/tensor.tcc"

#endif
