/* Trapzodial Numerical Integration */
void trapz(int *m, int *n, double *x, double *y, double *out);

/* Trapezoidal numerical integration in third dimension. */
void trapz3(int nr, int nc, int nd, double *x, double *y, double *z);

/* L2 Vector Norm */
void pvecnorm2(int *n, double *x, double *dt, double *out);
void pvecnorm(int *n, double *x, double *dt, double *out);

/* Gradient using forward and centered differences */
void gradient(int *m, int *n, double *f, double *binsize, double *g);

/* Computes gradient in across columns of f. */
void col_gradient(int nrows, int ncols, double *f, double step, double *df);

/* Matrix product operation (c = a*b) */
void product(int m, int n, int nn, double *a, double *b, double *c);

/* Cummulative Trapzodial Numerical Integration */
void cumtrapz(int *n, double *x, double *y, double *z);

/* Simpson Numerical Integration */
void simpson(int *m1, int *n1, double *x, double *y, double *out);

/* SRSF Inner Product */
void innerprod_q(int *m1, double *t, double *q1, double *q2, double *out);

/* SRVF Inner Product */
double innerprod_q2(int *m1, double *q1, double *q2);

/* 1-D Sample Covariance */
void cov(int n, double *x, double *y, double *out);

/* Spline Interpolation */
void spline(int n, double *x, double *y, int nu, double *xi, double *yi);
void spline_coef(int n, double *x, double *y, double *b, double *c, double *d);
void spline_eval(int nu, double *u, double *v, int n, double *x, double *y, double *b, double *c, double *d);

/* Linear Interpoloation */
void approx(double *x, double *y, int nxy, double *xout, double *yout, int nout, int method, double yleft, double yright, double f);

/* Invert Gamma */
void invertGamma(int n, double *gam, double *out);

/* SqrtMeanInverse - find proper inverse of mean of warping functions */
void SqrtMeanInverse(int *T1, int *n1, double *ti, double *gami, double *out);

/* linear spaced vector */
void linspace(double min, double max, int n, double *result);

/* reparameterize srvf q by gamma */
void group_action_by_gamma(int *n1, int *T1, double *q, double *gam, double *qn);
