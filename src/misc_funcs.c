#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

/* Structure of Linear Interpolation */
typedef struct {
    double ylow;
    double yhigh;
    double f1;
    double f2;
    int kind;
} appr_meth;


void trapz(int *m, int *n, double *x, double *y, double *out) {
    int k, j;
    double *yptr;

    yptr = y;
    for (k=0; k<*n; k++) {
        out[k] = 0.0;
        for (j=0; j<*m-1; j++)
            out[k] += (x[j+1]-x[j])*(yptr[j+1]+yptr[j])*0.5;
        yptr += *m;
    }
}


/* Trapezoidal numerical integration in third dimension. */
void trapz3(int nr, int nc, int nd, double *x, double *y, double *z) {
    /* x is [1 x nd]
     * y is [nr x nc x nd]
     * z is [nr x nc] */
    int k, j, i, ij;
    int nrc = nr*nc;

    for (i=0; i<nr; i++)
        for (j=0; j<nc; j++) {
            ij = i+j*nr;
            z[ij] = 0;
            for (k=0; k<nd-1; k++)
                z[ij] += (x[k+1]-x[k])*(y[ij+(k+1)*nrc]+y[ij+k*nrc]);
            z[ij] *= 0.5;
        }
}


void pvecnorm2(int *n, double *x, double *dt, double *out) {
    int k;
    *out = 0.0;

    for (k=0; k<*n; k++) {
        *out += x[k]*x[k];
    }
    *out = sqrt(*out * *dt);
}


void pvecnorm(int *n, double *x, double *dt, double *out) {
    int k;
    *out = 0.0;

    for (k=0; k<*n; k++) {
        *out += x[k]*x[k];
    }
    *out = sqrt(*out) * *dt;
}


void cov(int n, double *x, double *y, double *out){
    int k;
    double xmean, ymean, tmp, N;

    N = n;

    xmean = 0;
    ymean = 0;
    for (k=0; k<n; k++){
        xmean += x[k];
        ymean += y[k];
    }
    xmean = xmean/n;
    ymean = ymean/n;

    tmp = 0;
    for (k=0; k<n; k++)
        tmp += (x[k] - xmean) * (y[k] - ymean);
    tmp = tmp/N;
    *out = tmp;
}


void gradient(int *m, int *n, double *f, double *binsize, double *g) {
    int k, j;
    double *g_ptr, *f_ptr;

    /* Take forward differences on left and right edges */
    g_ptr = g; f_ptr = f;
    for (k=0; k<*n; k++) {
        g_ptr[0] = (f_ptr[1] - f_ptr[0])/ *binsize;
        g_ptr[*m-1] = (f_ptr[*m-1] - f_ptr[*m-2])/ *binsize;

        /* move to next column */
        g_ptr += *m;
        f_ptr += *m;
    }

    /* Take centered differences on interior points */
    g_ptr = g; f_ptr = f;
    for (k=0; k<*n; k++) {
        for (j=1; j<*m-1; j++)
            g_ptr[j] = (f_ptr[j+1]-f_ptr[j-1])/(2.0* *binsize);
        g_ptr += *m;
        f_ptr += *m;
    }
}

/* Computes gradient in across columns of f. */
void col_gradient(int nrows, int ncols, double *f, double step, double *df) {
    int k, j;
    int colN = nrows*(ncols-1);
    int colN1 = nrows*(ncols-2);

    for (k=0; k<nrows; k++) { /* loop over rows of f */
        /* Take forward differences on left and right edges */
        df[k] = (f[k+nrows] - f[k])/step;
        df[k+colN] = (f[k+colN] - f[k+colN1])/step;

        /* Take centered differences on interior points */
        for (j=1; j<ncols-1; j++) {
            df[k+j*nrows] = (f[k+(j+1)*nrows]-f[k+(j-1)*nrows])/(2*step);
        }
    }
}


/* Matrix product operation (c = a*b) */
void product(int m, int n, int nn, double *a, double *b, double *c) {
    /* a is [m x n]
     * b is [n x nn]
     * c is [m x nn] */
    int k, j, i;

    for (k=0; k<m; k++)
        for (j=0; j<nn; j++) {
            c[k+j*m] = 0; /* zero out c[k,j] */
            for (i=0; i<n; i++)
                c[k+j*m] += a[k+i*m]*b[i+j*n];
            }
}


void cumtrapz(int *n, double *x, double *y, double *z) {
    int k;
    z[0] = 0.0;
    for (k=1; k<*n; k++)
        z[k] = z[k-1] + 0.5*(y[k-1]+y[k])*(x[k]-x[k-1]);
}


void simpson(int *m1, int *n1, double *x, double *y, double *out) {
    int k, j;
    double *yptr;
    double dx1, dx2;
    double alpha, a0, a1, a2;
    int n = *n1;
    int m = *m1;

    if (m<3) {
        trapz(m1,n1,x,y,out);
    }
    else {
        for (k=0; k<n; k++)
        out[k] = 0.0;

        for (j=0; j<m-2; j=j+2) {
            dx1 = x[j+1] - x[j];
            dx2 = x[j+2] - x[j+1];

            alpha = (dx1+dx2)/dx1/6.0;
            a0 = alpha*(2.0*dx1-dx2);
            a1 = alpha*(dx1+dx2)*(dx1+dx2)/dx2;
            a2 = alpha*dx1/dx2*(2.0*dx2-dx1);

            yptr = y;
            for (k=0; k<n; k++) {
                out[k] += a0*yptr[j] + a1*yptr[j+1] +a2*yptr[j+2];
                yptr += m;
            }
        }

        if (m%2==0) {
            yptr = y;
            for (k=0; k<n; k++) {
                alpha = x[m-3]*x[m-2]*(x[m-3]-x[m-2]) -
                    x[m-3]*x[m-1]*(x[m-3]-x[m-1]) +
                    x[m-2]*x[m-1]*(x[m-2]-x[m-1]);
                a0 = yptr[m-3]*(x[m-2]-x[m-1]) - yptr[m-2]*(x[m-3]-x[m-1]) +
                    yptr[m-1]*(x[m-3]-x[m-2]);
                a1 = yptr[m-3]*(x[m-1]*x[m-1]-x[m-2]*x[m-2]) -
                    yptr[m-2]*(x[m-1]*x[m-1]-x[m-3]*x[m-3]) +
                    yptr[m-1]*(x[m-2]*x[m-2]-x[m-3]*x[m-3]);
                a2 = x[m-3]*x[m-2]*yptr[m-1]*(x[m-3]-x[m-2]) -
                    x[m-3]*yptr[m-2]*x[m-1]*(x[m-3]-x[m-1]) +
                    yptr[m-3]*x[m-2]*x[m-1]*(x[m-2]-x[m-1]);
                a0 /= alpha; a1 /= alpha; a2 /= alpha;

                out[k] += a0*(x[m-1]*x[m-1]*x[m-1]-x[m-2]*x[m-2]*x[m-2])/3 +
                    a1*(x[m-1]*x[m-1]-x[m-2]*x[m-2])/2 + a2*(x[m-1]-x[m-2]);
                yptr += m;
            }
        }
    }
}

/* SRSF Inner Product */
void innerprod_q(int *m1, double *t, double *q1, double *q2, double *out) {
    int k;
    double *q;
    int m = *m1;
    int n1 = 1;

    q = (double *) malloc(m*sizeof(double));
    for (k=0; k<m; k++)
        q[k] = q1[k]*q2[k];

    trapz(m1, &n1, t, q, out);

    free(q);
}

/* SRVF Inner Product */
double innerprod_q2(int *m1, double *q1, double *q2) {
    int k;
    double *q;
    int m = *m1;
    int n1 = 2;
    double out = 0.0;

    q = (double *) malloc((n1*m)*sizeof(double));
    for (k=0; k<n1*m; k++)
        q[k] = q1[k]*q2[k];

    for (k=0; k<n1*m; k++)
        out += q[k];

    out = out/m;

    free(q);

    return(out);
}


/*
 *  Splines a la Forsythe Malcolm and Moler
 *  ---------------------------------------
 *  In this case the end-conditions are determined by fitting
 *  cubic polynomials to the first and last 4 points and matching
 *  the third derivitives of the spline at the end-points to the
 *  third derivatives of these cubics at the end-points.
 */
void spline_coef(int n, double *x, double *y, double *b, double *c, double *d)
{
    int nm1, i;
    double t;

    /* Adjustment for 1-based arrays */

    x--; y--; b--; c--; d--;

    if(n < 3) {
    t = (y[2] - y[1]);
    b[1] = t / (x[2]-x[1]);
    b[2] = b[1];
    c[1] = c[2] = d[1] = d[2] = 0.0;
    return;
    }

    nm1 = n - 1;

    /* Set up tridiagonal system */
    /* b = diagonal, d = offdiagonal, c = right hand side */

    d[1] = x[2] - x[1];
    c[2] = (y[2] - y[1])/d[1];/* = +/- Inf  for x[1]=x[2] -- problem? */
    for(i=2 ; i<n ; i++) {
    d[i] = x[i+1] - x[i];
    b[i] = 2.0 * (d[i-1] + d[i]);
    c[i+1] = (y[i+1] - y[i])/d[i];
    c[i] = c[i+1] - c[i];
    }

    /* End conditions. */
    /* Third derivatives at x[0] and x[n-1] obtained */
    /* from divided differences */

    b[1] = -d[1];
    b[n] = -d[nm1];
    c[1] = c[n] = 0.0;
    if(n > 3) {
    c[1] = c[3]/(x[4]-x[2]) - c[2]/(x[3]-x[1]);
    c[n] = c[nm1]/(x[n] - x[n-2]) - c[n-2]/(x[nm1]-x[n-3]);
    c[1] = c[1]*d[1]*d[1]/(x[4]-x[1]);
    c[n] = -c[n]*d[nm1]*d[nm1]/(x[n]-x[n-3]);
    }

    /* Gaussian elimination */

    for(i=2 ; i<=n ; i++) {
    t = d[i-1]/b[i-1];
    b[i] = b[i] - t*d[i-1];
    c[i] = c[i] - t*c[i-1];
    }

    /* Backward substitution */

    c[n] = c[n]/b[n];
    for(i=nm1 ; i>=1 ; i--)
    c[i] = (c[i]-d[i]*c[i+1])/b[i];

    /* c[i] is now the sigma[i-1] of the text */
    /* Compute polynomial coefficients */

    b[n] = (y[n] - y[n-1])/d[n-1] + d[n-1]*(c[n-1]+ 2.0*c[n]);
    for(i=1 ; i<=nm1 ; i++) {
    b[i] = (y[i+1]-y[i])/d[i] - d[i]*(c[i+1]+2.0*c[i]);
    d[i] = (c[i+1]-c[i])/d[i];
    c[i] = 3.0*c[i];
    }
    c[n] = 3.0*c[n];
    d[n] = d[nm1];
    return;
}


void spline_eval(int nu, double *u, double *v, int n, double *x, double *y, double *b, double *c, double *d)
{
    /* Evaluate  v[l] := spline(u[l], ...),     l = 1,..,nu, i.e. 0:(nu-1)
    * Nodes x[i], coef (y[i]; b[i],c[i],d[i]); i = 1,..,n , i.e. 0:(*n-1)
    */

    const int n_1 = n - 1;
    int i, j, k, l;
    double ul, dx, tmp;
    int method = 3; // fmm

    if(method == 1 && n > 1) { /* periodic */
    dx = x[n_1] - x[0];
    for(l = 0; l < nu; l++) {
        v[l] = fmod(u[l]-x[0], dx);
        if(v[l] < 0.0) v[l] += dx;
        v[l] += x[0];
    }
    } else for(l = 0; l < nu; l++) v[l] = u[l];

    for(l = 0, i = 0; l < nu; l++) {
    ul = v[l];
    if(ul < x[i] || (i < n_1 && x[i+1] < ul)) {
        /* reset i  such that  x[i] <= ul <= x[i+1] : */
        i = 0;
        j = n;
        do {
        k = (i+j)/2;
        if(ul < x[k]) j = k;
        else i = k;
        } while(j > i+1);
    }
    dx = ul - x[i];
    /* for natural splines extrapolate linearly left */
    tmp = (method == 2 && ul < x[0]) ? 0.0 : d[i];

    v[l] = y[i] + dx*(b[i] + dx*(c[i] + dx*tmp));
    }
}


void spline(int n, double *x, double *y, int nu, double *xi, double *yi) {
    double *b = malloc(sizeof(double)*(n));
    double *c = malloc(sizeof(double)*(n));
    double *d = malloc(sizeof(double)*(n));

    spline_coef(n, x, y, b, c, d);
    spline_eval(nu, xi, yi, n, x, y, b, c, d);

    free(b); free(c); free(d);
    return;
}


static double approx1(double v, double *x, double *y, int n, appr_meth *Meth) {
  /* Approximate  y(v),  given (x,y)[i], i = 0,..,n-1 */
  int i, j, ij;

  i = 0;
  j = n - 1;

  /* handle out-of-domain points */
  if(v < x[i]) return Meth->ylow;
  if(v > x[j]) return Meth->yhigh;

  /* find the correct interval by bisection */
  while(i < j - 1) { /* x[i] <= v <= x[j] */
		ij = (i + j)/2; /* i+1 <= ij <= j-1 */
		if(v < x[ij]) j = ij; else i = ij;
		/* still i < j */
  }
  /* provably have i == j-1 */

  /* interpolation */

  if(v == x[j]) return y[j];
  if(v == x[i]) return y[i];
  /* impossible: if(x[j] == x[i]) return y[i]; */

  if(Meth->kind == 1) /* linear */
		return y[i] + (y[j] - y[i]) * ((v - x[i])/(x[j] - x[i]));
  else /* 2 : constant */
		return y[i] * Meth->f1 + y[j] * Meth->f2;
}


void approx(double *x, double *y, int nxy, double *xout, double *yout,
	    int nout, int method, double yleft, double yright, double f)
{
    int i;
    appr_meth M = {0.0, 0.0, 0.0, 0.0, 0}; /* -Wall */

    M.f2 = f;
    M.f1 = 1 - f;
    M.kind = method;
    M.ylow = yleft;
    M.yhigh = yright;
    for(i = 0; i < nout; i++)
			yout[i] = approx1(xout[i], x, y, nxy, &M);

		return;
}

void invertGamma(int n, double *gam, double *out) {
	double *x = malloc(sizeof(double)*(n));
	double *y = malloc(sizeof(double)*(n));
	int k;

	for (k=0; k<n; k++)
		x[k] = (double)k/((double)(n-1));

	approx(gam, x, n, x, out, n, 1, 0, 1, 0);
	out[n] = 1;

	for (k=0; k<n; k++)
		out[k] = out[k]/out[n];

	free(x); free(y);
	return;
}


void SqrtMeanInverse(int *T1, int *n1, double *ti, double *gami, double *out){
    int T = *T1, n = *n1;
    int maxiter = 30, tt = 1;
    const int size_array = T*n;
    double *psi = malloc(sizeof(double)*size_array);
    double *gam = malloc(sizeof(double)*size_array);
    double *mu = malloc(sizeof(double)*T);
    double *vec = malloc(sizeof(double)*size_array);
    double *v = malloc(sizeof(double)*T);
    double *y = malloc(sizeof(double)*T);
    double *vm = malloc(sizeof(double)*T);
    double *mnpsi = malloc(sizeof(double)*T);
    double *dqq = malloc(sizeof(double)*n);
    double *tmpv = malloc(sizeof(double)*T);
    double *gam_mu = malloc(sizeof(double)*T);
    double *lvm = malloc(sizeof(double)*maxiter);
    double tmpi, len;
    double eps = DBL_EPSILON, min = 0.0, binsize, tmp;
    int k, iter, l, n2 = 1, min_ind = 0;
   
    double *x = malloc(sizeof(double)*(T));

    for (k=0; k<maxiter; k++)
        lvm[k] = 0;

    for (k=0; k<T; k++)
        gam_mu[k] = 0;

    // remove
    for (k=0; k<T; k++)
        x[k] = (double)k/((double)(T-1));

    // pointers
    double *psi_ptr, *gam_ptr, *tmp1_ptr, *y_ptr, *gam_mu_ptr;

    binsize = 0;
    for (k=0; k<T-1; k++)
        binsize += ti[k+1]-ti[k];
    binsize = binsize/(T-1);

    for (k=0; k<T*n; k++){
        gam[k] = gami[k];
    }
    psi_ptr = psi; gam_ptr = gam;
    gradient(&T,&n,gam_ptr,&binsize,psi_ptr);
    for (k=0; k<T*n; k++)
        psi[k] = sqrt(fabs(psi[k])+eps);

    // Initilize
    for (k=0; k<T; k++){
        tmp = 0;
        for(l=0; l<n; l++)
            tmp += psi[k+l*T];

        mnpsi[k] = tmp/n;
    }
    for (k=0; k<n; k++){
        for(l=0; l<T; l++)
            tmpv[l] = psi[k*T+l]-mnpsi[l];
        tmp = 0;
        for(l=0; l<T; l++)
            tmp += tmpv[l];
        dqq[k] = tmp;
    }

    for (k=0; k<n; k++) {
        if (k==0) {
            min_ind = 0;
            min = dqq[k];
        }
        else{
            if (dqq[k]<min){
                min = dqq[k];
                min_ind = k;
            }
        }
    }

    for (k=0; k<T; k++)
        mu[k] = psi[(min_ind-1)*T+k];


    // Find Direction
    for (iter=1; iter<maxiter; iter++){
        for (k=0; k<n; k++){
            for(l=0; l<T; l++)
                v[l] = psi[k*T+l] - mu[l];

            for(l=0; l<T; l++)
                y[l] = psi[k*T+l]*mu[l];
            y_ptr = y;
            trapz(T1, &n2, ti, y_ptr, &tmpi);

            if (tmpi > 1){
                tmpi = 1;
            }
            else if (tmpi < (-1)){
                tmpi = 1;
            }

            len = acos(tmpi);
            if (len > 0.0001){
                for(l=0; l<T; l++)
                    vec[k*T+l] = (len/sin(len))*(psi[k*T+l]-cos(len)*mu[l]);
            }
            else {
                for(l=0; l<T; l++)
                    vec[k*T+l] = 0;
            }

        }

        for (k=0; k<T; k++){
            tmp = 0;
            for(l=0; l<n; l++){
                tmp += vec[k+l*T];
            }
            vm[k] = tmp/n;
        }

        for (k=0; k<T; k++)
            tmpv[k] = vm[k]*vm[k];

        tmp = 0;
        for (k=0; k<T; k++)
            tmp += tmpv[k];

        lvm[iter] = sqrt(tmp*binsize);

        if (lvm[iter] == 0){
            break;
        }

        for (k=0; k<T; k++)
            mu[k] = cos(tt*lvm[iter])*mu[k]+(sin(tt*lvm[iter])/lvm[iter])*vm[k];

        if (lvm[iter] < 1e-6 || iter >= maxiter)
            break;
    }

    for (k=0; k<T; k++)
        tmpv[k] = mu[k]*mu[k];

    tmp1_ptr = tmpv;
    gam_mu_ptr = gam_mu;
    cumtrapz(T1,ti,tmp1_ptr,gam_mu_ptr);
    for (k=0; k<T; k++)
        gam_mu_ptr[k] = (gam_mu_ptr[k] - gam_mu_ptr[0])/(gam_mu_ptr[T-1]-gam_mu_ptr[0]); // slight change of scale

    invertGamma(T, gam_mu_ptr, out);

    free(psi);
    free(gam);
    free(mu);
    free(vec);
    free(v);
    free(y);
    free(vm);
    free(mnpsi);
    free(dqq);
    free(tmpv);
    free(gam_mu);
    free(lvm);
    free(x);
    return;
}


/* linear spaced vector */
void linspace(double min, double max, int n, double *result){
    int iterator = 0;
    int i;
    double temp;

    for (i = 0; i <= n-2; i++){
        temp = min + i*(max-min)/(floor((double)n) - 1);
        result[iterator] = temp;
        iterator += 1;
     }

    result[iterator] = max;

    return;
}


/* reparameterize srvf q by gamma */
void group_action_by_gamma(int *n1, int *T1, double *q, double *gam, double *qn){
    int T = *T1, n = *n1;
	int n2 = 1;
    double dt = 1.0/T, max=1, min=0;
    int j, k;
    double val;
    double *gammadot = malloc(sizeof(double)*T);
    double *ti = malloc(sizeof(double)*T);
    double *tmp = malloc(sizeof(double)*T);
    double *tmp1 = malloc(sizeof(double)*T);
    double *gammadot_ptr, *time_ptr, *tmp_ptr, *tmp1_ptr;

    time_ptr = ti;
    linspace(min, max, T, time_ptr);
    gammadot_ptr = gammadot;
    gradient(T1,&n2,gam,&dt,gammadot_ptr);

    for (k=0; k<n; k++){
		tmp_ptr = tmp;
		tmp1_ptr = tmp1;
        for (j=0; j<T; j++)
            tmp[j] = q[n*j+k];
        spline(T, time_ptr, tmp_ptr, T, gam, tmp1_ptr);
        for (j=0; j<T; j++)
            qn[n*j+k] = tmp1[j]* sqrt(gammadot[j]);

    }

    val = innerprod_q2(T1, qn, qn);

    for (k=0; k<T*n; k++)
        qn[k] = qn[k] / sqrt(val);
    

    free(gammadot);
    free(ti);
    free(tmp);
    free(tmp1);

    return;
}
