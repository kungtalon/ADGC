#ifndef ADGC_UTILS_THREADS_H_
#define ADGC_UTILS_THREADS_H_

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace utils {
namespace threads {

// ThreadPool creates a bunch of threads and manages multi-thread tasks
class ThreadPool {
  /*
  Credit: https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
  */
public:
  ThreadPool(){};
  ~ThreadPool();
  void start(size_t max_thread_num);
  void submit_job(const std::function<void()> &job);
  void stop();
  bool busy();

private:
  void thread_loop();

  bool should_terminate_ = false; // Tells threads to stop looking for jobs
  std::mutex queue_mutex_;        // Prevents data races to the job queue
  std::condition_variable
      mutex_condition_; // Allows threads to wait on new jobs or termination
  std::vector<std::thread> threads_;
  std::queue<std::function<void()>> jobs_;
};

// ConcurrentCounter not used
template <typename K> class ConcurrentCounter {
public:
  ~ConcurrentCounter();
  size_t get(K *key);
  size_t increment(K *key);
  size_t decrement(K *key);

private:
  void destroy(K *key);
  void destroy_all();
  std::unordered_map<K *, size_t> data_;
  mutable std::shared_mutex mutex_; // Prevents data races to the job queue
};

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

} // namespace threads
} // namespace utils

#endif
