cmake_minimum_required(VERSION 3.15)
project(adgc)

# Options. whether use tests
option(DO_TEST "DO_TEST" ON)
option(USE_GRAPHVIZ "USE_GRAPHVIZ" ON)

# global setting
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pthread)

add_definitions(-DADGC_MULTI_THREADS_NUM_=3)

if (USE_GRAPHVIZ)
  add_definitions(-DADGC_ENABLE_GRAPHVIZ_)
endif()

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
    "src/utils/math_utils.tcc"
    "include/utils/thread.h"
    "src/utils/thread.cc"
    "include/utils/utils.h"
    "src/utils/utils.tcc"
)
set_target_properties(utils_lib PROPERTIES LINKER_LANGUAGE CXX)

if (USE_GRAPHVIZ)
add_library(graph_utils_lib
  "include/utils/graph_utils.h"
)
endif()
set_target_properties(graph_utils_lib PROPERTIES LINKER_LANGUAGE CXX)

find_package( BLAS REQUIRED )
if(BLAS_FOUND)
  message("BLAS found.")
endif(BLAS_FOUND)

add_library(tensor_lib
  "include/tensor/tensor.h"
  "src/tensor/tensor.tcc"
  "include/tensor/mapper.h"
  "src/tensor/mapper.tcc"
)
set_target_properties(tensor_lib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tensor_lib  
  ${BLAS_LIBRARIES} 
)

add_library(graph_core_lib
    "include/autodiff/consts.h"
    "include/autodiff/node.h"
    "include/autodiff/graph.h"
    "include/autodiff/variable.h"
    "src/autodiff/node.cc"
    "src/autodiff/graph.cc"
    "src/autodiff/variable.cc"
)
if(USE_GRAPHVIZ)
  target_link_libraries(graph_core_lib
    cgraph
    gvc
  )
endif(USE_GRAPHVIZ)

add_library(ops_funcs_lib
  "include/autodiff/ops.h"
  "include/autodiff/functional.h"
  "src/autodiff/ops.cc"
  "src/autodiff/ops_impl.cc"
  "src/autodiff/functional.cc"
  "src/autodiff/functional_impl.cc"
)
target_link_libraries(ops_funcs_lib
  graph_core_lib
)

add_library(layer_lib
  "include/autodiff/layer/layers.h"
  "include/autodiff/layer/dense.h"
  "src/autodiff/layer/layers.cc"
  "src/autodiff/layer/dense.cc"
)

target_link_libraries(layer_lib  
  ops_funcs_lib
  tensor_lib
  utils_lib
)

# testing
if(DO_TEST)
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
    utils_lib
    gtest
    gtest_main
    gmock
    gmock_main
  )

  if(USE_GRAPHVIZ)
    add_executable(
      graph_test
      test/graph_test.cc
    )
    target_link_libraries(graph_test 
      ops_funcs_lib
      graph_utils_lib
      tensor_lib
      utils_lib
      gtest
      gtest_main
      gmock
      gmock_main
    )
  endif()

  add_executable(
    ops_funcs_test
    test/ops_funcs_test.cc
  )
  target_link_libraries(ops_funcs_test 
    ops_funcs_lib
    tensor_lib
    utils_lib
    gtest
    gtest_main
    gmock
    gmock_main
  )

  add_executable(
    layer_test
    test/layer_test.cc
  )
  target_link_libraries(layer_test 
    layer_lib
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
  if(USE_GRAPHVIZ)
    add_test(graph_test graph_test)
  endif()
  add_test(ops_funcs_test ops_funcs_test)
  add_test(layer_test layer_test)
endif()