#include "MLP.h"

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double learning_rate) {
  this->layers = layers;
  this->bias = bias;
  this->learning_rate = learning_rate;

  for (int layer_i = 0; layer_i < this->layers.size(); layer_i++) {
    this->values.push_back(vector<double>(layers[layer_i], 0.0));  // init all SUM_wi.xi to 0
    this->network.push_back(vector<Perceptron>());

    if (layer_i > 0) {  // network[0] == input layer => no neurons
      for (int j = 0; j < layers[layer_i]; j++) {
        this->network[layer_i].push_back(Perceptron(layers[layer_i - 1], bias));
      }
    }
  }
}

void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> w_init) {
  for (int i = 0; i < w_init.size(); i++)
    for (int j = 0; j < w_init[i].size(); j++)
      this->network[i + 1][j].set_weights(w_init[i][j]);
}
void MultiLayerPerceptron::print_weights() {
  cout << endl;
  for (int i = 1; i < this->network.size(); i++) {
    for (int j = 0; j < this->layers[i]; j++) {
      cout << "Layer " << i + 1 << " Neuron " << j + 1 << ": ";
      for (auto &it : network[i][j].getWeights()) {
        cout << it << "\t\t";
      }
      cout << endl;
    }
  }
  cout << endl;
}
vector<double> MultiLayerPerceptron::predict(vector<double> x) {
  this->values[0] = x;
  for (int i = 1; i < this->network.size(); i++)
    for (int j = 0; j < this->layers[i]; j++)
      this->values[i][j] = this->network[i][j].predict(this->values[i - 1]);
  return this->values.back();
}
