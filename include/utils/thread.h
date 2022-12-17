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

} // namespace threads
} // namespace utils

#include "utils/thread.cc"

#endif
