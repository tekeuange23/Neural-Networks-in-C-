#pragma once
#include <time.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

/**
 *   x0     ---------|
 *   x1     ---------|
 *    .              |
 *    .              ===----> z = SUM wi.xi  ---->  activate(z) ------
 *    .              \
 *  x(n-1)  ---------|
 *  b == xn ---------|
 */
class Perceptron {
 private:
  /* data */
  vector<double> weights;
  double bias;

 public:
  Perceptron(int inputs, double bias = 1.0);
  // ~Perceptron();

  /* getters & setters */
  vector<double> getWeights();
  void set_weights(vector<double> w_init);

  /* methods */
  double predict(vector<double> x);
  double sigmoid(double x);
};

double squareRoot(const double a);