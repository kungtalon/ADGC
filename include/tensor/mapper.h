#ifndef ADGC_UTILS_MAPPER_H_
#define ADGC_UTILS_MAPPER_H_

#include <functional>
#include <vector>

#include "utils/thread.h"

// Mapper is the class used to instantiate a function
// applied to every entry in the tensor

#define ADGC_MAX_MAPPER_THREADS_ 3

namespace tensor {

template <typename dType> class Mapper {
public:
  Mapper(){};
  Mapper(const std::function<void(dType &)> &func);
  Mapper(const std::function<void(dType &, const size_t &)> &func);

  void run(dType *p_tensor, const size_t &size,
           utils::threads::ThreadPool *p_pool = nullptr);

private:
  std::function<void(dType &, const size_t &)> task_;
};

} // namespace tensor

#include "tensor/mapper.tcc"

#endif