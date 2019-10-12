#include "armadillo"
#include "bayesian.h"

using namespace arma;
using namespace std;

// calculate exponential map in f.exp1()
vec calcY(double area, vec gy) {
  int len = gy.size();
  double sarea = sin(area);
  double carea = cos(area);
  vec output(len);
  if (area != 0) {
    for (int i = 0; i < len; i++) {
      output[i] = carea + sarea / area * gy[i];
    }
    return output;
  } else {
    for (int i = 0; i < len; i++) {
      output[i] = 1.0;
    }
    return output;
  }
}


// cumulative sum of squares for f.SSEg.pw()
vec cuL2norm2(vec x, vec y) {
  int n = x.size();

  // get ordering of x
  // borrowed from Hadley Wickhams github.com/hadley/adv-r/blob/master/extras/cpp/order.cpp
  
  vector<pair<double, int> > vals;
  vals.reserve(n);
  for(int i = 0; i < n; i++) {
    vals.push_back(make_pair(x[i], i));
  }

  std::sort(vals.begin(), vals.end());
  int sInd;
  vec xSort(n);
  vec ySortSq (n);
  for(int i = 0; i < n; i++) {
    sInd = vals[i].second;
    xSort[i] = x[sInd];
    ySortSq[i] = pow(y[sInd], 2.0);
  }

  // Get trapezoid areas
  vec prod_xy2(n-1);
  for (int i = 0; i < (n-1); i++) {
    prod_xy2[i] = (xSort[i+1] - xSort[i]) * (ySortSq[i+1] + ySortSq[i]);
  }

  // cumulative sum
  vec cusum(n-1);
  for (int i = 0; i < (n-1); i++) {
    cusum[i] = 0;
    for (int j = 0; j <= i; j++) {
      cusum[i] += prod_xy2[j];
    }
  }
  return (cusum / 2.0);
}

// simple trapezoidal numerical integration
double trapzCpp(vec x, vec y) {
  int n = x.size();
  double area2 = 0.0;
  for (int i = 0; i < (n-1); i++) {
    area2 += (x[i+1] - x[i]) * (y[i+1] + y[i]);
  }
  return (area2/2.0);
}

// order vectors and calculate l2 norm, for f.L2norm()
double order_l2norm(vec x, vec y) {
  int n = x.size();
  // get ordering of x
  // borrowed from Hadley Wickhams adv-r/extras/cpp/order.cpp

  vector<pair<double, int> > vals;
  vals.reserve(n);
  for(int i = 0; i < n; i++) {
    vals.push_back(make_pair(x[i], i));
  }

  std::sort(vals.begin(), vals.end());
  int sInd;
  vec xSort(n);
  vec ySortSq(n);
  for(int i = 0; i < n; i++) {
    sInd = vals[i].second;
    xSort[i] = x[sInd];
    ySortSq[i] = pow(y[sInd], 2.0);
  }

  // loop through sorted inds, square y, trapz
  double area2 = 0.0;
  for (int i = 0; i<(n-1); i++) {
    area2 += (xSort[i+1] - xSort[i]) * (ySortSq[i+1] + ySortSq[i]);
  }
  
  double out = sqrt(area2/2.0);
  return (out);
}
