#ifndef BAYESIAN_H
#define BAYESIAN_H

#include "armadillo"

using namespace arma;
using namespace std;

vec calcY(double area, vec gy);

vec cuL2norm2(vec x, vec y); 

// simple trapezoidal numerical integration
double trapzCpp(vec x, vec y); 

// order vectors and calculate l2 norm, for f.L2norm()
double order_l2norm(vec x, vec y);

#endif // end of BAYESIAN_H
