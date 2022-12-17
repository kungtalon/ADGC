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

} // namespace threads
} // namespace utils