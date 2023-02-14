#include <iostream>
#include <cstring>
#include <cmath>

using namespace std;

#define SQRT5	2.23606797749978969641
#define EPS	0.00000000001

// Prototypes: -----------------------------------------------------------------
double InProd(const double *u, const double *v, int d);
void GramSchmitd(double *x, int &n, int d);

double innerSquare(const double *u, const double *v, int n, int t, int d);
void GramSchmitdSquare(double *x, int &n, int n1, int n2, int d);
