#include <gtest/gtest.h>

#include "../MLP.cpp"

// TEST(SquareRootTest, PositiveNos) {
//   ASSERT_EQ(6, squareRoot(36.0));
//   ASSERT_EQ(18.0, squareRoot(324.0));
//   ASSERT_EQ(25.4, squareRoot(645.16));
//   ASSERT_EQ(0, squareRoot(0.0));
// }

/************************************************************
                        LOGICAL AND GATE
*************************************************************/
TEST(AND_GATE, test_the_case_of_specific_data_) {
  Perceptron* p = new Perceptron(2);
  p->set_weights({10, 10, -15});

  vector<vector<vector<double>>> matrix = {
      {{3.059 * pow(10, -7), precision(p->predict({0, 0}), 10)}, {6.69285 * pow(10, -3), precision(p->predict({0, 1}), 8)}},
      {{6.69285 * pow(10, -3), precision(p->predict({0, 1}), 8)}, {-0.214748 * pow(10, 0), precision(p->predict({1, 1}), 9)}}};
  // AND
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      cout << i << " - " << j << "\n";
      ASSERT_EQ(matrix[i][j][0], matrix[i][j][1]);
    }
  }
  // ASSERT_EQ(3.059 * pow(10, -7), precision(p->predict({0, 0}), 10));
  // ASSERT_EQ(p->predict({0, 1}), 0);
  // ASSERT_EQ(p->predict({1, 0}), 0);
  // ASSERT_EQ(p->predict({1, 1}), 0);
}
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}