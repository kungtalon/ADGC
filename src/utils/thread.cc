#include "utils/thread.h"

namespace utils {
namespace threads {

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::start(size_t max_thread_num) {
  const size_t system_thread_num =
      std::thread::hardware_concurrency(); // Max # of threads the system
                                           // supports
  size_t num_threads = system_thread_num;
  if (max_thread_num > 0 && system_thread_num > max_thread_num) {
    num_threads = max_thread_num;
  }

  threads_.resize(num_threads);
  for (int i = 0; i < num_threads; i++) {
    threads_.at(i) = std::thread(&ThreadPool::thread_loop, this);
  }
}

void ThreadPool::submit_job(const std::function<void()> &job) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    jobs_.push(job);
  }
  mutex_condition_.notify_one();
}

void ThreadPool::stop() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    should_terminate_ = true;
  }
  mutex_condition_.notify_all();
  for (std::thread &active_thread : threads_) {
    active_thread.join();
  }
  threads_.clear();
}

bool ThreadPool::busy() {
  bool poolbusy;
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    poolbusy = !jobs_.empty();
  }
  return poolbusy;
}

void ThreadPool::thread_loop() {
  while (true) {
    std::function<void()> job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      mutex_condition_.wait(
          lock, [this] { return !jobs_.empty() || should_terminate_; });
      if (should_terminate_) {
        return;
      }
      job = jobs_.front();
      jobs_.pop();
    }
    job();
  }
}

template <typename K> ConcurrentCounter<K>::~ConcurrentCounter() {
  destroy_all();
}

template <typename K> size_t ConcurrentCounter<K>::get(K *key) {
  std::shared_lock lock(mutex_);
  auto result_it = data_.find(key);
  if (result_it == data_.end()) {
    return 0;
  } else {
    return result_it->second;
  }
}

template <typename K> size_t ConcurrentCounter<K>::increment(K *key) {
  std::unique_lock lock(mutex_);
  auto result_it = data_.find(key);
  if (result_it == data_.end()) {
    data_[key] = 1;
    return 1;
  } else {
    return ++data_[key];
  }
}

template <typename K> size_t ConcurrentCounter<K>::decrement(K *key) {
  std::unique_lock lock(mutex_);
  auto result_it = data_.find(key);
  if (result_it == data_.end()) {
    std::cerr << "Decrementing count of an non-existent pointer!!" << std::endl;
    return 0;
  } else {
    size_t cur_count = --data_[key];
    if (cur_count == 0) {
      destroy(key);
    }
    return cur_count;
  }
}

template <typename K> void ConcurrentCounter<K>::destroy(K *key) {
  // called by decrement, do not acquire lock here
  auto result_it = data_.find(key);
  if (result_it == data_.end()) {
    std::cerr << "ConcurrentCounter : Attempting to destroy an "
                 "non-existent pointer!!"
              << std::endl;
  } else {
    if (data_[key] != 0) {
      std::cerr << "ConcurrentCounter : Attempting to destroy a "
                   "referenced pointer!!"
                << std::endl;
      return;
    }
    delete[] key;
  }
  data_.erase(result_it);
  key = nullptr;
}

template <typename K> void ConcurrentCounter<K>::destroy_all() {
  if (!data_.empty()) {
    std::unique_lock lock(mutex_);
    for (auto it = data_.begin(); it != data_.end(); it++) {
      delete[] it->first;
    }
    data_.clear();
  }
}

ConcurrentCounter<int32_t> inst_int_concurrent_counter;
ConcurrentCounter<float> inst_float_concurrent_counter;
ConcurrentCounter<double> inst_double_concurrent_counter;

} // namespace threads
} // namespace utils