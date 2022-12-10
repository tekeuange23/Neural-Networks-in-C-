#include <gtest/gtest.h>
#include <string.h>

#include "../MLP.cpp"
#include "../Perceptron.cpp"

/*********************************************  LOGICAL AND GATE PREDICTION  **********************************************
 *                                              LINEAR                          ___________________________________________
 *           _________                                                          | A | B |  z  |        Y        |   A^B   |
 *   A   --->|w0 = 10|****|                                                     +---+---+-----+-----------------+---------+
 *           _________    **|                                                   | 0 | 0 | -15 | 3.059 * 10^ -7  |    0    |
 *   B   --->|w1 = 10|------|===----> z = SUM wi.xi  ---->  Y = Sigmoid(z)      +---+---+-----+-----------------+---------+
 *           _________    __|                                                   | 0 | 1 | -5  | 6.692 * 10^ -3  |    0    |
 * b = 1 --->|w2= -15|____|                                                     +---+---+-----+-----------------+---------+
 *           *********                                                          | 1 | 0 | -5  | 6.692 * 10^ -3  |    0    |
 *                                                                              +---+---+-----+-----------------+---------+
 *                                                                              | 1 | 1 | +5  |    0.993307     |    1    |
 **************************************************************************************************************************/
TEST(AND_GATE, _______________USING_A_TWO_INPUT_PERCEPTRON_WITH_DEFINED_WEIGHT_______________) {
  Perceptron* p = new Perceptron(2);
  p->set_weights({10, 10, -15});

  vector<vector<vector<double>>> matrix = {
      {{3.059 * pow(10, -7), precision(p->predict({0, 0}), 10)}, {6.69285 * pow(10, -3), precision(p->predict({0, 1}), 8)}},
      {{6.69285 * pow(10, -3), precision(p->predict({0, 1}), 8)}, {0.993307 * pow(10, 0), precision(p->predict({1, 1}), 9)}}};

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      // cout << i << " - " << j << "\n";
      ASSERT_EQ(to_string(matrix[i][j][0]), to_string(matrix[i][j][1]));
    }
  }
}

/*********************************************  LOGICAL OR GATE PREDICTION   **********************************************
 *                                              LINEAR                          ___________________________________________
 *           _________                                                          | A | B |  z  |        Y        |   AvB   |
 *   A   --->|w0 = 10|****|                                                     +---+---+-----+-----------------+---------+
 *           _________    **|                                                   | 0 | 0 | -5  |    0.00669285   |    0    |
 *   B   --->|w1 = 10|------|===----> z = SUM wi.xi  ---->  Y = Sigmoid(z)      +---+---+-----+-----------------+---------+
 *           _________    __|                                                   | 0 | 1 | +5  |     0.993307    |    1    |
 * b = 1 --->|w2 = -5|____|                                                     +---+---+-----+-----------------+---------+
 *           *********                                                          | 1 | 0 | +5  |     0.993307    |    1    |
 *                                                                              +---+---+-----+-----------------+---------+
 *                                                                              | 1 | 1 | +15 |         1       |    1    |
 **************************************************************************************************************************/
TEST(OR_GATE, ________________USING_A_TWO_INPUT_PERCEPTRON_WITH_DEFINED_WEIGHT_______________) {
  Perceptron* p = new Perceptron(2);
  p->set_weights({10, 10, -5});

  vector<vector<vector<double>>> matrix = {
      {{0.00669285, precision(p->predict({0, 0}), 10)}, {0.993307, precision(p->predict({0, 1}), 8)}},
      {{0.993307, precision(p->predict({0, 1}), 8)}, {1, precision(p->predict({1, 1}), 9)}}};

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ASSERT_EQ(to_string(matrix[i][j][0]), to_string(matrix[i][j][1]));
    }
  }
}

/*********************************************  LOGICAL XOR GATE PREDICTION  **********************************************
 *                                              Xor(A,B) = And(Or(A,B), NAnd(A,B))
 *                                              :NON-LINEAR                           _____________________________________
 *   A   --->___________                                                              | A | B |        Y        | Xor(A,B)|
 *   B   --->| OR_GATE |****|                                                         +---+---+-----------------+---------+
 * b = 1 --->___________    ------------>__________                                   | 0 | 0 |     0.007153    |    0    |
 *                          ------------>|AND_GATE|_______Y___                        +---+---+-----------------+---------+
 *   A   --->___________    |    b=1 --->__________                                   | 0 | 1 |     0.992356    |    1    |
 *   B   --->|NAND_GATE|____|                                                         +---+---+-----------------+---------+
 * b = 1 --->___________                                                              | 1 | 0 |     0.992356    |    1    |
 *                                                                                    +---+---+-----------------+---------+
 *                                                                                    | 1 | 1 |     0.007153    |    0    |
 **************************************************************************************************************************/
TEST(XOR_GATE, __________________USING_A_TWO_INPUT_MLP_WITH_DEFINED_WEIGHT___________________) {
  MultiLayerPerceptron* mlp = new MultiLayerPerceptron({2, 2, 1});
  mlp->set_weights({{{-10, -10, 15}, {10, 10, -5}},
                    {{10, 10, -15}}});

  mlp->print_weights();

  vector<vector<vector<double>>> matrix = {
      {{0.007153, precision(mlp->predict({0, 0})[0], 10)}, {0.992356, precision(mlp->predict({0, 1})[0], 8)}},
      {{0.992356, precision(mlp->predict({0, 1})[0], 8)}, {0.007153, precision(mlp->predict({1, 1})[0], 9)}}};

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      // cout << i << " - " << j << "\n";
      ASSERT_EQ(to_string(matrix[i][j][0]), to_string(matrix[i][j][1]));
    }
  }
}

/*********************************************   LOGICAL XOR GATE TRAINING   **********************************************
 *                                              Xor(A,B) = And(Or(A,B), NAnd(A,B))
 *                                              :NON-LINEAR                           _____________________________________
 *   A   --->___________                                                              | A | B |        Y        | Xor(A,B)|
 *   B   --->| OR_GATE |****|                                                         +---+---+-----------------+---------+
 * b = 1 --->___________    ------------>__________                                   | 0 | 0 |     0.007153    |    0    |
 *                          ------------>|AND_GATE|_______Y___                        +---+---+-----------------+---------+
 *   A   --->___________    |    b=1 --->__________                                   | 0 | 1 |     0.992356    |    1    |
 *   B   --->|NAND_GATE|____|                                                         +---+---+-----------------+---------+
 * b = 1 --->___________                                                              | 1 | 0 |     0.992356    |    1    |
 *                                                                                    +---+---+-----------------+---------+
 *                                                                                    | 1 | 1 |     0.007153    |    0    |
 **************************************************************************************************************************/
TEST(XOR_GATE, ___________________________TRAINING_A_TWO_INPUT_MLP___________________________) {
  MultiLayerPerceptron* mlp = new MultiLayerPerceptron({2, 2, 1});

  /** TRAINING **/
  double MSE;
  int epochs = 3000;

  for (int i = 0; i < epochs; i++) {
    MSE = 0.0;

    cout << "The MSE after " << i << "\ttraining epochs equals to " << MSE << endl;
    MSE = mlp->back_propagate({0, 0}, {0});
    cout << "The MSE after " << i << "\ttraining epochs equals to " << MSE << endl;
    MSE = mlp->back_propagate({0, 1}, {1});
    MSE = mlp->back_propagate({1, 0}, {1});
    MSE = mlp->back_propagate({1, 1}, {0});
    MSE = MSE / 4.0;
    if (i % 100 == 0) {
      cout << "The MSE after " << i << "\ttraining epochs equals to " << MSE << endl;
    }
  }

  mlp->print_weights();

  vector<vector<vector<double>>> matrix = {
      {{0, mlp->predict({0, 0})[0]}, {1, mlp->predict({0, 1})[0]}},
      {{1, mlp->predict({0, 1})[0]}, {0, mlp->predict({1, 1})[0]}}};

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      // cout << i << " - " << j << "\n";
      ASSERT_EQ(to_string(matrix[i][j][0]), to_string(matrix[i][j][1]));
    }
  }
}

/**************************************************************************************************************************/
/*************************************************   MAIN TESTS RUNNER  ***************************************************/
/**************************************************************************************************************************/
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}