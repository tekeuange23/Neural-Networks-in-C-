#include "Perceptron.h"

class MultiLayerPerceptron {
 public:
  MultiLayerPerceptron(vector<int>, layers, double bias = 1.0, double eta = 0.5);
  void set_weights(vector<vector<vector<double>>> w_init);
  void print_weights();
  vector<double> predict(vector<double> x);
  double bp(vector<double> x, vector<double> y);
}
