#include "MLP.h"

/* Helper's prototypes */
double frand();

/***************************************************************************************
                              Constructors | Getters | Setters
****************************************************************************************/
/**
 * @inputs: number of input of our perceptron;
 * @bias: bias of the perceptron;
 * @return a new Perceptron obj with inputs + 1 (for the bias) inputs;
 */
Perceptron::Perceptron(int inputs, double bias) {
  this->bias = bias;
  this->weights.resize(inputs + 1);
  generate(weights.begin(), weights.end(), frand);
}
vector<double> Perceptron::getWeights() {
  return this->weights;
}
void Perceptron::setWeights(vector<double> w_init) {
  this->weights = w_init;
}

/***************************************************************************************
                                         Helpers
****************************************************************************************/
/**
 * @return a random value between -1 and 1;
 */
double frand() {
  double time2 = 2.0 * (double)rand() / RAND_MAX;  // time2 belongs to [0, 2]
  return time2 - 1.0;                              // belongs to [-1, 1]
}
double squareRoot(const double a) {
  double b = sqrt(a);
  if (b != b) {  // nan check
    return -1.0;
  } else {
    return sqrt(a);
  }
}
/***************************************************************************************
                                         Methods
****************************************************************************************/
double Perceptron::predict(vector<double> x) {
  x.push_back(this->bias);
  double z = inner_product(x.begin(), x.end(), this->weights.begin(), 0.0);
  return sigmoid(z);
}
double Perceptron::sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}