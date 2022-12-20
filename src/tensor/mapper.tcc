#include "tensor/mapper.h"

namespace tensor {

template <typename dType>
Mapper<dType>::Mapper(const std::function<void(dType &)> &func) {
  task_ = [&func](dType &value, const size_t &index) { func(value); };
}

template <typename dType>
Mapper<dType>::Mapper(const std::function<void(dType &, const size_t &)> &func)
    : task_(func) {}

template <typename dType>
void Mapper<dType>::run(dType *p_tensor, const size_t &size,
                        utils::threads::ThreadPool *p_pool) {
  if (p_pool != nullptr) {
    for (size_t ix = 0; ix < size; ++ix) {
      p_pool->submit_job(
          [this, &p_tensor, ix] { this->task_(*(p_tensor + ix), ix); });
    }
    while (true) {
      if (!p_pool->busy()) {
        break;
      }
    }

    p_pool->stop();
  } else {
    for (size_t ix = 0; ix < size; ++ix) {
      task_(*(p_tensor + ix), ix);
    }
  }
}

} // namespace tensor
