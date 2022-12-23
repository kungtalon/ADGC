cmake_minimum_required(VERSION 3.15)
project(adgc)

# global setting
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -pthread)

# Options. whether use tests
option(SKIP_TEST "whether to skip ctest" OFF)
option(USE_GRAPHVIZ "whether to use graphviz" ON)

add_definitions(-DADGC_MULTI_THREADS_NUM_=3)

if (${SKIP_TEST})
  message("-- [INFO] Skipped all tests...")
endif()

if (${USE_GRAPHVIZ})
  add_definitions(-DADGC_ENABLE_GRAPHVIZ_)
  message("-- [INFO] Using graphviz...")
else()
  message("-- [INFO] Graphviz is disabled...")
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

file(GLOB UTILS_LIB_FILES "include/utils/*.h" "src/utils/*cc")
add_library(utils_lib ${UTILS_LIB_FILES})
set_target_properties(utils_lib PROPERTIES LINKER_LANGUAGE CXX)

if (${USE_GRAPHVIZ})
add_library(graph_utils_lib
  "include/utils/graph_utils.h"
)
set_target_properties(graph_utils_lib PROPERTIES LINKER_LANGUAGE CXX)
endif()

find_package( BLAS REQUIRED )
if(BLAS_FOUND)
  message("BLAS found.")
endif(BLAS_FOUND)


add_library(tensor_lib
  "include/tensor/tensor.h"
  "src/tensor/tensor.tcc"
  "src/tensor/tensor_impl.cc"
  "include/tensor/mapper.h"
  "src/tensor/mapper.tcc"
)
set_target_properties(tensor_lib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tensor_lib  
  ${BLAS_LIBRARIES} 
)

add_library(graph_core_lib
    "include/autodiff/consts.h"
    "include/autodiff/graph.h"
    "include/autodiff/component/node.h"
    "include/autodiff/component/variable.h"
    "src/autodiff/graph.cc"
    "src/autodiff/component/node.cc"
    "src/autodiff/component/variable.cc"
)
if(${USE_GRAPHVIZ})
  target_link_libraries(graph_core_lib
    cgraph
    gvc
  )
endif()

add_library(graph_component_lib
  "include/autodiff/component/ops.h"
  "include/autodiff/component/functional.h"
  "src/autodiff/component/ops.cc"
  "src/autodiff/component/ops_impl.cc"
  "src/autodiff/component/functional.cc"
  "src/autodiff/component/functional_impl.cc"
)
target_link_libraries(graph_component_lib
  graph_core_lib
)

file(GLOB LAYER_LIB_FILES "include/autodiff/layer/*.h" "src/autodiff/layer/*.cc")
add_library(layer_lib ${LAYER_LIB_FILES})
target_link_libraries(layer_lib  
  graph_component_lib
  tensor_lib
  utils_lib
)

file(GLOB OPTIMIZER_LIB_FILES "include/autodiff/optimizer/*.h" "src/autodiff/optimizer/*.cc")
add_library(optimizer_lib ${OPTIMIZER_LIB_FILES})
target_link_libraries(optimizer_lib  
  graph_component_lib
  tensor_lib
  utils_lib
)

file(GLOB DATA_LIB_FILES "include/data/*.h" "src/data/*.cc")
add_library(data_lib ${DATA_LIB_FILES})
set_target_properties(data_lib PROPERTIES LINKER_LANGUAGE CXX)



# testing
if(NOT ${SKIP_TEST})
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

  if(${USE_GRAPHVIZ})
    add_executable(
      graph_test
      test/graph_test.cc
    )
    target_link_libraries(graph_test 
      graph_component_lib
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
    graph_component_lib
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

  add_executable(
    optimizer_test
    test/optimizer_test.cc
  )
  target_link_libraries(optimizer_test 
    layer_lib
    optimizer_lib
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
  if(${USE_GRAPHVIZ})
    add_test(graph_test graph_test)
  endif()
  add_test(ops_funcs_test ops_funcs_test)
  add_test(layer_test layer_test)
  add_test(optimizer_test optimizer_test)
endif()