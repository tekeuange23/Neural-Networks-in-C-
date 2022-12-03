#include "MLP.h"

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double learning_rate) {
  this->layers = layers;
  this->bias = bias;
  this->learning_rate = learning_rate;

  for (int layer_i = 0; layer_i < this->layers.size(); layer_i++) {
    this->value.push_back(vector<double>(layers[i], 0.0));  // init network related values to 0
    this->network.push_back(vector<Perceptron>());

    if (layer_i > 0) {  // network[0] == input layer => no neurons
      for (int j = 0; j < layer_i; j++) {
        this->network[layer_i].push_back(Perceptron(this->network[layer_i - 1], bias));
      }
    }
  }
}

void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> w_init) {
}
void MultiLayerPerceptron::print_weights() {
}