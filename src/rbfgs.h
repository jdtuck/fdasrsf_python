#ifndef RBFGS_H
#define RBFGS_H

#include "armadillo"

using namespace arma;
using namespace std;

vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty);

#endif // end of RBFGS_H
