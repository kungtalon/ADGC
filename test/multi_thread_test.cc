#include "utils/thread.h"
#include "gtest/gtest.h"

TEST(AdgcMuliThreadTest, MultiThreadMap) {
  std::vector<int> nums = std::vector<int>(11);
  try {
    auto thread_pool = utils::threads::ThreadPool();
    size_t max_thread_num = 1;
    thread_pool.start(max_thread_num);

    for (int ix = 1; ix <= 10; ix++) {
      thread_pool.submit_job([&nums, ix]() { nums[ix] = ix * ix; });
    }

    while (true) {
      if (!thread_pool.busy()) {
        break;
      }
    }
  } catch (std::exception &ex) {
    FAIL() << "Get exception " << ex.what();
  }
  for (int ix = 1; ix <= 10; ix++) {
    EXPECT_EQ(nums[ix], ix * ix)
        << "Wrong result at index " << ix << " , got " << nums[ix];
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}