#include <iostream>

#include "autodiff/variable.h"
#include "gtest/gtest.h"

typedef graph_component::Graph g;
typedef graph_component::Variable v;

TEST(GraphTest, ConstructorDestructorTest) {
  g *graph_ptr = new g();

  // use nested code block to make sure graph_ptr outlives node_ptr
  {
    v *node_ptr0 = new v({2});
    v *node_ptr1 = new v({2}, {}, "node1", true, true, graph_ptr);
    v *node_ptr2 = new v({2}, {node_ptr1}, "node2", true, true, graph_ptr);
    v *node_ptr3 =
        new v({2}, {node_ptr1, node_ptr2}, "node3", true, true, graph_ptr);

    EXPECT_THROW(new v({2}, {node_ptr1, node_ptr2}),
                 adg_exception::MismatchRegisterdGraphError);
    EXPECT_THROW(new v({2}, {}, "node3", true, true, graph_ptr),
                 adg_exception::DuplicateNodeNameError);
  }
  delete graph_ptr;
  SUCCEED();
}

#ifdef ADGC_ENABLE_GRAPHVIZ_
TEST(GraphTest, GraphVizTest) {
  g *graph_ptr = new g();

  // use nested code block to make sure graph_ptr outlives node_ptr
  {
    v *node_ptr1 = new v({2}, graph_ptr);
    v *node_ptr2 = new v({2}, {node_ptr1}, graph_ptr);
    v *node_ptr3 =
        new v({2}, {node_ptr1, node_ptr2}, "custom", true, true, graph_ptr);
    v *node_ptr4 = new v({2}, {node_ptr1}, "", true, true, graph_ptr);
    v *node_ptr5 =
        new v({2}, {node_ptr1, node_ptr4}, "", true, true, graph_ptr);

    graph_ptr->visualize("../graphviz/out.svg");
  }
  delete graph_ptr;
  SUCCEED();
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}