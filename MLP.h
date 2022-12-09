#include "Perceptron.h"

class MultiLayerPerceptron {
 public:
  MultiLayerPerceptron(vector<int> layers, double bias = 1.0, double learning_rate = 0.5);
  // ~MultiLayerPerceptron();

  void set_weights(vector<vector<vector<double>>> w_init);
  void print_weights();
  vector<double> predict(vector<double> x);
  double back_propagate(vector<double> x, vector<double> y);

 private:
  double bias;
  double learning_rate;

  vector<vector<Perceptron>> network;
  vector<int> layers;             // number of neurons per layers including the input layer
  vector<vector<double>> values;  // SUM_wi.xi outputs values of each perceptron
  vector<vector<double>> delta;   // error term
};