cmake_minimum_required(VERSION 3.15)

# Options. whether use tests
option(test "Build all tests." OFF)

# global setting
set(CMAKE_CXX_STANDARD 20)
project(kungtalon_adgc)

set(CMAKE_CXX_FLAGS -pthread)

include_directories("${PROJECT_SOURCE_DIR}/include/"
                    "${PROJECT_SOURCE_DIR}/test/"
                    "${PROJECT_SOURCE_DIR}/src/")

# gTest module
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# global libraries
# file(GLOB UTILS_SOURCES "src/utils/*.cc" "src/utils/*.tcc" "include/utils/*.h")
# add_library(utils SHARED ${UTILS_SOURCES})
add_library(utils_lib
  "include/utils/math_utils.h"
  "include/utils/math_utils.h"
  "include/utils/thread.h"
  "include/utils/utils.h"
)
set_target_properties(utils_lib PROPERTIES LINKER_LANGUAGE CXX)

find_package( BLAS REQUIRED )
if(BLAS_FOUND)
    message("BLAS found.")
endif(BLAS_FOUND)

add_library(tensor_lib
  "include/autodiff/tensor.h"
  "src/autodiff/tensor.tcc"
)
set_target_properties(tensor_lib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tensor_lib  
  ${BLAS_LIBRARIES} 
  utils_lib
)


# testing
enable_testing()

add_executable(
  multi_thread_test
  test/multi_thread_test.cc
)
target_link_libraries(multi_thread_test
  utils_lib
  gtest
  gtest_main
)

add_executable(
  math_test
  test/math_test.cc
)
target_link_libraries(math_test 
  ${BLAS_LIBRARIES} 
  utils_lib
  gtest
  gtest_main
)

add_executable(
  utils_test
  test/utils_test.cc
)
target_link_libraries(utils_test 
  utils_lib
  gtest
  gtest_main
)


add_executable(
  tensor_test
  test/tensor_test.cc
)
target_link_libraries(tensor_test 
  tensor_lib
  gtest
  gtest_main
  gmock
  gmock_main
)

include(GoogleTest)
add_test(multi_thread_test multi_thread_test)
add_test(utils_test utils_test)
add_test(math_test math_test)
add_test(tensor_test tensor_test)