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
      for (auto &it : network[i][j].get_weights()) {
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
double MultiLayerPerceptron::back_propagate(vector<double> x, vector<double> y) {
  // STEP 1: Feed a sample to the network
  vector<double> outputs = this->predict(x);

  // STEP 2: Calculate the MSE
  vector<double> error;
  double MSE = 0.0;

  for (int i = 0; i < y.size(); i++) {
    error.push_back(y[i] - outputs[i]);
    MSE += pow(error[i], 2);
  }
  MSE /= this->layers.back();

  /**
   * TODO:
   * why not divide by --> [y.size()]
   */

  // STEP 3: Calculate the output error terms
  cout << "------------------------OKKKKK " << y.size() << " == " << outputs.size() << endl;
  for (int i = 0; i < outputs.size(); i++) {
    this->delta.back()[i] = outputs[i] * (1 - outputs[i]) * error[i];
  }

  // STEP 4: Calculate the error term of each unit on each layer
  for (int i = this->network.size() - 2; i > 0; i--) {
    for (int h = 0; h < this->network[i].size(); h++) {
      double forward_error = 0.0;
      for (int k = 0; k < this->layers[i + 1]; k++) {
        forward_error += this->network[i + 1][k].get_weights()[h] * this->delta[i + 1][k];
      }
      this->delta[i][h] = this->values[i][h] * (1 - this->values[i][h]) * forward_error;
    }
  }

  // STEP 5 & 6 : Calculate the deltas and update the weights
  for (int i = 1; i < this->network.size(); i++) {
    for (int j = 0; j < this->layers[i]; j++) {
      for (int k = 0; k < this->layers[i - 1] + 1; k++) {  // number of inputs (#neurons in the prev layer) +1 for the bias input neuron
        double delta_;
        if (k == this->layers[i - 1]) {
          delta_ = this->learning_rate * this->delta[i][j] * this->bias;
        } else {
          delta_ = this->learning_rate * this->delta[i][j] * this->values[i - 1][k];
        }
        this->network[i][j].incr_weight(k, delta_);
      }
    }
  }

  return MSE;
}