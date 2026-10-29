#ifndef RBFGS_H
#define RBFGS_H

#include "armadillo"

using namespace arma;
using namespace std;

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty);
// gam0, when not empty, is the warping to start from instead of the identity;
// info is filled with [iterations, stop reason, cost, gradient norm], the stop
// reason being 1 for a vanishing step, 2 for the gradient norm and 3 for
// maxiter
vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty,
                 vec gam0, vec& info);

#endif // end of RBFGS_H
