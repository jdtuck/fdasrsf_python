#include <cstring>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include "UnitSquareImage.h"
#include "ImageRegister.h"

using namespace std;

void findgrad(double *dfdu, double *dfdv, const double *f, int n, int t) {
    int i, j, k, N = n*t;
    double du, dv;

    du = 1.0/(t-1);
    dv = 1.0/(n-1);

    for (i = 0; i < 1; ++i) {
        // j = 0, k = 0
        dfdu[n*(t*i + 0) + 0] = fdiff(f + n*(t*i + 0) + 0, du, n);
        dfdv[n*(t*i + 0) + 0] = fdiff(f + n*(t*i + 0) + 0, dv, 1);

        // k = 0
        for (j = 1; j < n-1; ++j) {
            dfdu[n*(t*i + 0) + j] = fdiff(f + n*(t*i + 0) + j, du, n);
            dfdv[n*(t*i + 0) + j] = cdiff(f + n*(t*i + 0) + j, dv, 1);
        }

        // j = n-1, k = 0
        dfdu[n*(t*i + 0) + n-1] = fdiff(f + n*(t*i + 0) + n-1, du, n);
        dfdv[n*(t*i + 0) + n-1] = bdiff(f + n*(t*i + 0) + n-1, dv, 1);

        for (k = 1; k < t-1; ++k) {
            // j = 0
            dfdu[n*(t*i + k) + 0] = cdiff(f + n*(t*i + k) + 0, du, n);
            dfdv[n*(t*i + k) + 0] = fdiff(f + n*(t*i + k) + 0, dv, 1);

            for (j = 1; j < n-1; ++j) {
                dfdu[n*(t*i + k) + j] = cdiff(f + n*(t*i + k) + j, du, n);
                dfdv[n*(t*i + k) + j] = cdiff(f + n*(t*i + k) + j, dv, 1);
            }

            // j = n-1
            dfdu[n*(t*i + k) + n-1] = cdiff(f + n*(t*i + k) + n-1, du, n);
            dfdv[n*(t*i + k) + n-1] = bdiff(f + n*(t*i + k) + n-1, dv, 1);
        }

        // j = 0, k = t-1
        dfdu[n*(t*i + t-1) + 0] = bdiff(f + n*(t*i + t-1) + 0, du, n);
        dfdv[n*(t*i + t-1) + 0] = fdiff(f + n*(t*i + t-1) + 0, dv, 1);

        // k = t-1
        for (j = 1; j < n-1; ++j) {
            dfdu[n*(t*i + t-1) + j] = bdiff(f + n*(t*i + t-1) + j, du, n);
            dfdv[n*(t*i + t-1) + j] = cdiff(f + n*(t*i + t-1) + j, dv, 1);
        }

        // j = n-1, k = t-1
        dfdu[n*(t*i + t-1) + n-1] = bdiff(f + n*(t*i + t-1) + n-1, du, n);
        dfdv[n*(t*i + t-1) + n-1] = bdiff(f + n*(t*i + t-1) + n-1, dv, 1);
    }

}

// ------------------------------------------------------------------------
void findgrad2D(double *dfdu, double *dfdv, double *f, int n, int t, int d) {
    int i, j, k, N = n*t;
    double du, dv;

    du = 1.0/(t-1);
    dv = 1.0/(n-1);

    for (i = 0; i < d; ++i) {
        // j = 0, k = 0
        dfdu[n*(t*i + 0) + 0] = fdiff(f + n*(t*i + 0) + 0, du, n);
        dfdv[n*(t*i + 0) + 0] = fdiff(f + n*(t*i + 0) + 0, dv, 1);

        // k = 0
        for (j = 1; j < n-1; ++j) {
            dfdu[n*(t*i + 0) + j] = fdiff(f + n*(t*i + 0) + j, du, n);
            dfdv[n*(t*i + 0) + j] = cdiff(f + n*(t*i + 0) + j, dv, 1);
        }

        // j = n-1, k = 0
        dfdu[n*(t*i + 0) + n-1] = fdiff(f + n*(t*i + 0) + n-1, du, n);
        dfdv[n*(t*i + 0) + n-1] = bdiff(f + n*(t*i + 0) + n-1, dv, 1);

        for (k = 1; k < t-1; ++k) {
            // j = 0
            dfdu[n*(t*i + k) + 0] = cdiff(f + n*(t*i + k) + 0, du, n);
            dfdv[n*(t*i + k) + 0] = fdiff(f + n*(t*i + k) + 0, dv, 1);

            for (j = 1; j < n-1; ++j) {
                dfdu[n*(t*i + k) + j] = cdiff(f + n*(t*i + k) + j, du, n);
                dfdv[n*(t*i + k) + j] = cdiff(f + n*(t*i + k) + j, dv, 1);
            }

            // j = n-1
            dfdu[n*(t*i + k) + n-1] = cdiff(f + n*(t*i + k) + n-1, du, n);
            dfdv[n*(t*i + k) + n-1] = bdiff(f + n*(t*i + k) + n-1, dv, 1);
        }

        // j = 0, k = t-1
        dfdu[n*(t*i + t-1) + 0] = bdiff(f + n*(t*i + t-1) + 0, du, n);
        dfdv[n*(t*i + t-1) + 0] = fdiff(f + n*(t*i + t-1) + 0, dv, 1);

        // k = t-1
        for (j = 1; j < n-1; ++j) {
            dfdu[n*(t*i + t-1) + j] = bdiff(f + n*(t*i + t-1) + j, du, n);
            dfdv[n*(t*i + t-1) + j] = cdiff(f + n*(t*i + t-1) + j, dv, 1);
        }

        // j = n-1, k = t-1
        dfdu[n*(t*i + t-1) + n-1] = bdiff(f + n*(t*i + t-1) + n-1, du, n);
        dfdv[n*(t*i + t-1) + n-1] = bdiff(f + n*(t*i + t-1) + n-1, dv, 1);
    }

}

// ------------------------------------------------------------------------
void multfact_image(double *multfact, const double *dfdu, const double *dfdv, int n, int t,int d) {
    int N = n*t;

    //     multfact = new double[N];

    if (d < 3) {
        for (int i = 0; i < N; ++i) {
            multfact[i] = abs(dfdu[N*0+i]*dfdv[N*1+i] - dfdu[N*1+i]*dfdv[N*0+i]);
        }
    }
    else if (d < 4) {
        for (int i = 0; i < N; ++i) {
            multfact[i] = pow(pow(dfdu[N*0+i]*dfdv[N*1+i] - dfdu[N*1+i]*dfdv[N*0+i],2) +
                    pow(dfdu[N*0+i]*dfdv[N*2+i] - dfdu[N*2+i]*dfdv[N*0+i],2) +
                    pow(dfdu[N*1+i]*dfdv[N*2+i] - dfdu[N*2+i]*dfdv[N*1+i],2), 1/2);
        }
    }
    else {
        for (int i = 0; i < N; ++i) {
            multfact[i] = pow(pow(dfdu[N*0+i]*dfdv[N*1+i] - dfdu[N*1+i]*dfdv[N*0+i],2) +
                    pow(dfdu[N*0+i]*dfdv[N*2+i] - dfdu[N*2+i]*dfdv[N*0+i],2) +
                    pow(dfdu[N*0+i]*dfdv[N*3+i] - dfdu[N*3+i]*dfdv[N*0+i],2) +
                    pow(dfdu[N*1+i]*dfdv[N*2+i] - dfdu[N*2+i]*dfdv[N*1+i],2) +
                    pow(dfdu[N*1+i]*dfdv[N*3+i] - dfdu[N*3+i]*dfdv[N*1+i],2) +
                    pow(dfdu[N*2+i]*dfdv[N*3+i] - dfdu[N*3+i]*dfdv[N*2+i],2), 1/2);
        }
    }
}

// ------------------------------------------------------------------------
void surface_to_q(double *q, const double *f, const double *multfact, int n, int t, int d) {
    int N = n*t;

    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < N; ++i) {
            q[N*k + i] = sqrt(multfact[i])*f[N*k + i];
        }
    }
}

// ------------------------------------------------------------------------
void Calculate_Distance(double *H, const double *q1, const double *q2, int n, int t, int d) {
    int N = n*t*d;
    double tmp, du, dv;

    du = 1.0/(n-1);
    dv = 1.0/(t-1);

    *H = 0;

    for (int i = 0; i < N; ++i) {
        tmp = q1[i] - q2[i];
        *H += tmp*tmp;
    }

    *H *= du*dv;
}

// ------------------------------------------------------------------------
void findphistar(double *w, double *q, double *b, int n, int t, int d, int K) {
    int D = 2;
    double du, dv, dbxdu, dbydv, divb, *dqdu, *dqdv, *expr1, *expr2;

    du = 1.0/(t-1);
    dv = 1.0/(n-1);

    expr1 = new double[d];
    expr2 = new double[d];

    dqdu = new double[n*t*d];
    dqdv = new double[n*t*d];

    findgrad2D(dqdu, dqdv, q, n, t, d);

    memset(w,0,n*t*d*K*sizeof(double));

    for (int j = 0; j < K; ++j) {
        // k = 0, i = 0
        dbxdu = fdiff(b + n*(t*(D*j + 0) + 0) + 0, du, n);
        dbydv = fdiff(b + n*(t*(D*j + 1) + 0) + 0, dv, 1);
        divb = 0.5*(dbxdu + dbydv);

        for (int ii = 0; ii < d; ++ii) {
            expr1[ii] = divb*q[n*(t*ii + 0) + 0];
            expr2[ii] = dqdu[n*(t*ii + 0) + 0]*b[n*(t*(D*j + 0) + 0) + 0] + dqdv[n*(t*ii + 0) + 0]*b[n*(t*(D*j + 1) + 0) + 0];
            w[n*(t*(d*j + ii) + 0) + 0] = expr1[ii] + expr2[ii];
        }

        // k = 0
        for (int i = 1; i < n-1; ++i) {
            dbxdu = fdiff(b + n*(t*(D*j + 0) + 0) + i, du, n);
            dbydv = cdiff(b + n*(t*(D*j + 1) + 0) + i, dv, 1);
            divb = 0.5*(dbxdu + dbydv);

            for (int ii = 0; ii < d; ++ii) {
                expr1[ii] = divb*q[n*(t*ii + 0) + i];
                expr2[ii] = dqdu[n*(t*ii + 0) + i]*b[n*(t*(D*j + 0) + 0) + i] + dqdv[n*(t*ii + 0) + i]*b[n*(t*(D*j + 1) + 0) + i];
                w[n*(t*(d*j + ii) + 0) + i] = expr1[ii] + expr2[ii];
            }
        }

        // i = n-1, k = 0
        dbxdu = fdiff(b + n*(t*(D*j + 0) + 0) + n-1, du, n);
        dbydv = bdiff(b + n*(t*(D*j + 1) + 0) + n-1, dv, 1);
        divb = 0.5*(dbxdu + dbydv);

        for (int ii = 0; ii < d; ++ii) {
            expr1[ii] = divb*q[n*(t*ii + 0) + n-1];
            expr2[ii] = dqdu[n*(t*ii + 0) + n-1]*b[n*(t*(D*j + 0) + 0) + n-1] + dqdv[n*(t*ii + 0) + n-1]*b[n*(t*(D*j + 1) + 0) + n-1];
            w[n*(t*(d*j + ii) + 0) + n-1] = expr1[ii] + expr2[ii];
        }

        for (int k = 1; k < t-1; ++k) {
            // i = 0
            dbxdu = cdiff(b + n*(t*(D*j + 0) + k) + 0, du, n);
            dbydv = fdiff(b + n*(t*(D*j + 1) + k) + 0, dv, 1);
            divb = 0.5*(dbxdu + dbydv);

            for (int ii = 0; ii < d; ++ii) {
                expr1[ii] = divb*q[n*(t*ii + k) + 0];
                expr2[ii] = dqdu[n*(t*ii + k) + 0]*b[n*(t*(D*j + 0) + k) + 0] + dqdv[n*(t*ii + k) + 0]*b[n*(t*(D*j + 1) + k) + 0];
                w[n*(t*(d*j + ii) + k) + 0] = expr1[ii] + expr2[ii];
            }

            for (int i = 1; i < n-1; ++i) {
                dbxdu = cdiff(b + n*(t*(D*j + 0) + k) + i, du, n);
                dbydv = cdiff(b + n*(t*(D*j + 1) + k) + i, dv, 1);
                divb = 0.5*(dbxdu + dbydv);

                for (int ii = 0; ii < d; ++ii) {
                    expr1[ii] = divb*q[n*(t*ii + k) + i];
                    expr2[ii] = dqdu[n*(t*ii + k) + i]*b[n*(t*(D*j + 0) + k) + i] + dqdv[n*(t*ii + k) + i]*b[n*(t*(D*j + 1) + k) + i];
                    w[n*(t*(d*j + ii) + k) + i] = expr1[ii] + expr2[ii];
                }
            }

            // i = n-1
            dbxdu = cdiff(b + n*(t*(D*j + 0) + k) + n-1, du, n);
            dbydv = bdiff(b + n*(t*(D*j + 1) + k) + n-1, dv, 1);
            divb = 0.5*(dbxdu + dbydv);

            for (int ii = 0; ii < d; ++ii) {
                expr1[ii] = divb*q[n*(t*ii + k) + n-1];
                expr2[ii] = dqdu[n*(t*ii + k) + n-1]*b[n*(t*(D*j + 0) + k) + n-1] + dqdv[n*(t*ii + k) + n-1]*b[n*(t*(D*j + 1) + k) + n-1];
                w[n*(t*(d*j + ii) + k) + n-1] = expr1[ii] + expr2[ii];
            }
        }

        // i = 0, k = t-1
        dbxdu = bdiff(b + n*(t*(2*j + 0) + t-1) + 0, du, n);
        dbydv = fdiff(b + n*(t*(2*j + 1) + t-1) + 0, dv, 1);
        divb = 0.5*(dbxdu + dbydv);

        for (int ii = 0; ii < d; ++ii) {
            expr1[ii] = divb*q[n*(t*ii + t-1) + 0];
            expr2[ii] = dqdu[n*(t*ii + t-1) + 0]*b[n*(t*(D*j + 0) + t-1) + 0] + dqdv[n*(t*ii + t-1) + 0]*b[n*(t*(D*j + 1) + t-1) + 0];
            w[n*(t*(d*j + ii) + t-1) + 0] = expr1[ii] + expr2[ii];
        }

        // k = t-1
        for (int i = 1; i < n-1; ++i) {
            dbxdu = bdiff(b + n*(t*(D*j + 0) + t-1) + i, du, n);
            dbydv = cdiff(b + n*(t*(D*j + 1) + t-1) + i, dv, 1);
            divb = 0.5*(dbxdu + dbydv);

            for (int ii = 0; ii < d; ++ii) {
                expr1[ii] = divb*q[n*(t*ii + t-1) + i];
                expr2[ii] = dqdu[n*(t*ii + t-1) + i]*b[n*(t*(D*j + 0) + t-1) + i] + dqdv[n*(t*ii + t-1) + i]*b[n*(t*(D*j + 1) + t-1) + i];
                w[n*(t*(d*j + ii) + t-1) + i] = expr1[ii] + expr2[ii];
            }
        }

        // i = n-1, k = t-1
        dbxdu = bdiff(b + n*(t*(D*j + 0) + t-1) + n-1, du, n);
        dbydv = bdiff(b + n*(t*(D*j + 1) + t-1) + n-1, dv, 1);
        divb = 0.5*(dbxdu + dbydv);

        for (int ii = 0; ii < d; ++ii) {
            expr1[ii] = divb*q[n*(t*ii + t-1) + n-1];
            expr2[ii] = dqdu[n*(t*ii + t-1) + n-1]*b[n*(t*(D*j + 0) + t-1) + n-1] + dqdv[n*(t*ii + t-1) + n-1]*b[n*(t*(D*j + 1) + t-1) + n-1];
            w[n*(t*(d*j + ii) + t-1) + n-1] = expr1[ii] + expr2[ii];
        }
    }

    delete [] dqdu;
    delete [] dqdv;
    delete [] expr1;
    delete [] expr2;

    return;
}


// ------------------------------------------------------------------------
void findupdategam(double *gamupdate, const double *v, const double *w, const double *b, int n, int t, int d, int K) {
    int i, k, N = n*t, D = 2;
    double innp, du, dv;

    du = 1.0/(n-1);
    dv = 1.0/(t-1);

    memset(gamupdate,0,n*t*D*sizeof(double));

    for (k = 0; k < K; ++k) {
        innp = 0;
        for (i = 0; i < N*d; ++i) {
            innp += v[i]*w[N*d*k + i];
        }

        innp *= du*dv;

        for (i = 0; i < N; ++i) {
            gamupdate[N*0 + i] += innp*b[N*D*k + N*0 + i];
            gamupdate[N*1 + i] += innp*b[N*D*k + N*1 + i];
        }
    }

    return;
}

// ------------------------------------------------------------------------
void updategam(double *gamnew, const double *gamupdate, const double *gamid, double eps, int n, int t, int D) {

    for (int i = 0; i < n*t*D; ++i) {
        gamnew[i] = gamid[i] + eps*gamupdate[i];
    }

    return;
}

// ------------------------------------------------------------------------
void Apply_gam_gamid(double *gamcum, const double *gamid, const double *gaminc, int m, int n) {
    int k;
    double u0, v0, ndu, ndv, t, *y, *D1, *D2;

    u0 = 0;
    v0 = 0;
    ndu = 1;
    ndv = 1;

    y = new double[n];
    D1 = new double[n];
    D2 = new double[m];

    for (int i = 0; i < m; ++i) { // for each row (i-th)
        for (int j = 0; j < n; ++j) {
            y[j] = gamid[m*(n*0+j)+i];
        } // y = (i-th) row of gamid[N*0]

        spline(D1, y, n);

        for (int j = 0; j < n; ++j) {
            lookupspline(t, k, gaminc[m*(n*0+j)+i]-u0, ndu, n);
            gamcum[m*(n*0+j)+i] = evalspline(t, D1+k, y+k);
        }
    }

    for (int j = 0; j < n; ++j) {
        spline(D2, gamid + m*(n*1+j)+0, m); // for each column (j-th)

        for (int i = 0; i < m; ++i) {
            lookupspline(t, k, gaminc[m*(n*1+j)+i]-v0, ndv, m);
            gamcum[m*(n*1+j)+i] = evalspline(t, D2+k, gamid + m*(n*1+j)+k);
        }
    }

    delete [] y;
    delete [] D1;
    delete [] D2;

    return;
}

//-------------------------------------------------------------------------
void Apply_Gamma_Surf(double *Fnew, const double *F, const double *gam, int m, int n, int d) {
    int j, N = m*n;
    double *Du, *Dv, *zu, u0, v0, ndu, ndv, u, v;

    Dv = new double[N];
    Du = new double[n];
    zu = new double[n];

    ndu = 1;
    ndv = 1;
    u0 = 0;
    v0 = 0;

    for (int k = 0; k < d; ++k) {
        interp2(Dv, F+(N*k+0), m, n); //col

        for (int i = 0; i < N; ++i) {
            lookupspline(v, j, gam[N*1+i]-v0, ndv, m); // col
            evalinterp2(v, Du, zu, Dv+j, F+(N*k+j), m, n); // row

            lookupspline(u, j, gam[N*0+i]-u0, ndu, n); // row
            Fnew[N*k + i] = evalspline(u, Du+j, zu+j); // row
        }
    }

    delete [] Dv;
    delete [] Du;
    delete [] zu;

    return;
}

// ------------------------------------------------------------------------
int check_crossing(double *f, int n, int t, int D) {
    int is_diffeo = 1;
    int N = n*t, nC = 0;
    double c, *dfdu, *dfdv;

    dfdu = new double[n*t*D];
    dfdv = new double[n*t*D];

    findgrad2D(dfdu,dfdv,f,n,t,D);

    for (int i=0; i<N; ++i) {
        c = dfdu[N*0 + i]*dfdv[N*1 + i] - dfdu[N*1 + i]*dfdv[N*0 + i];

        if (c < 0) {
            nC += 1;
        }
    }

    if (nC > 0)
        is_diffeo = 0;

    delete [] dfdu;
    delete [] dfdv;

    return is_diffeo;
}

//-------------------------------------------------------------------------
int ReparamSurf(double *Fnew, double *gamnew, double *H,
        double *Ft, const double *Fm, const double *gam,
        double *b, const double *gamid,
        const int n, const int t, const int d, const int D, const int K,
        double eps=0.1, const double tol=1e-8, const int itermax=100)
{
    int iter = 0, N = n*t, is_diffeo;
    double *qt, *qm, *v, *w, *gamupdate, *gaminc, *gamold, *Fu, *Fv, *multfact, Hdiff;

    qt = new double[n*t*d];
    qm = new double[n*t*d];
    v = new double[n*t*d];
    w = new double[n*t*d*K];
    gamupdate = new double[n*t*D];
    gaminc = new double[n*t*D];
    gamold = new double[n*t*D];
    Fu = new double[n*t*d];
    Fv = new double[n*t*d];
    multfact = new double[n*t];

    // initial gampold and Fold
    cpyArray(gamold,gam,n*t*D);
    Apply_Gamma_Surf(Fnew,Fm,gamold,n,t,d);

    // check point
    cpyArray(gamnew,gamold,n*t*D);

    // initial q-functions for Ft and Fnew
    findgrad2D(Fu,Fv,Ft,n,t,d);
    multfact_image(multfact,Fu,Fv,n,t,d);
    surface_to_q(qt,Ft,multfact,n,t,d);

    findgrad2D(Fu,Fv,Fnew,n,t,d);
    multfact_image(multfact,Fu,Fv,n,t,d);
    surface_to_q(qm,Fnew,multfact,n,t,d);

    // compute initial energy
    Hdiff = 100;
    Calculate_Distance(H+iter,qt,qm,n,t,d);

    // main iteration
    for (iter = 1; iter < itermax && Hdiff > tol; ++iter) {
        // basis to tangent space
        findphistar(w,qm,b,n,t,d,K);

        // find v = q1-q2
        for (int i = 0; i < N*d; ++i) {
            v[i] = qt[i]-qm[i];
        }

        findupdategam(gamupdate,v,w,b,n,t,d,K);

update:
        updategam(gaminc,gamupdate,gamid,eps,n,t,D);

        // compute update gampnew
        Apply_gam_gamid(gamnew,gamold,gaminc,n,t);
        is_diffeo = check_crossing(gamnew,n,t,D);
        if (!is_diffeo) {
            eps *= 0.5;
            goto update;
        }

        Apply_Gamma_Surf(Fnew,Fm,gamnew,n,t,d); // apply cumulative deformation to original object

        findgrad2D(Fu,Fv,Fnew,n,t,d);
        multfact_image(multfact,Fu,Fv,n,t,d);
        surface_to_q(qm,Fnew,multfact,n,t,d);

        Calculate_Distance(H+iter,qt,qm,n,t,d);
        Hdiff = (H[iter-1]-H[iter])/H[iter-1];

        // update iteration or break out
        if (H[iter] <= H[iter-1]) {
            cpyArray(gamold,gamnew,n*t*D);
        }
        else {
            eps *= 0.5;
            goto update;
        }
    }

    delete [] qt;
    delete [] qm;
    delete [] v;
    delete [] w;
    delete [] gamupdate;
    delete [] gaminc;
    delete [] gamold;
    delete [] Fu;
    delete [] Fv;
    delete [] multfact;

    return iter;
}

// inverse transform ------------------------------------------------------
void InvtGamma(double *gaminv, const double *gam, const double *gamid, int n) {
    int i, j, k;
    double ndu, ndv, t, *y, *D;

    ndu = 1;
    ndv = 1;

    y = new double[n];
    D = new double[n];

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            y[j] = gam[n*(n*0+j)+i];
        }

        spline(D, y, n);

        for (j = 0; j < n; ++j) {
            lookupspline(t, k, gamid[n*(n*0+j)+i], ndu, n);
            gaminv[n*(n*0+j)+i] = evalspline(t, D+k, y+k);
        }
    }

    for (j = 0; j < n; ++j) {
        spline(D, gam + n*(n*1+j)+0, n);

        for (i = 0; i < n; ++i) {
            lookupspline(t, k, gamid[n*(n*1+j)+i], ndv, n);
            gaminv[n*(n*1+j)+i] = evalspline(t, D+k, gam + n*(n*1+j)+k);
        }
    }

    delete [] y;
    delete [] D;

    return;
}

// ------------------------------------------------------------------------
void jacob_image(double *A, const double *F, int n, int t) {
    int j, k, N = n*t;
    double dfdu[2], dfdv[2], du, dv;
    double c=0.0;
    
    du = 1.0/(n-1);
    dv = 1.0/(t-1);

    // j = 0, k = 0
    fdiff2(dfdu, F + n*(t*0 + 0) + 0, du, n, N);
    fdiff2(dfdv, F + n*(t*0 + 0) + 0, dv, 1, N);
    jacob(c,dfdu,dfdv);
    A[n*0 + 0] = sqrt(abs(c));

    // k = 0
    for (j = 1; j < n-1; ++j) {
        fdiff2(dfdu, F + n*(t*0 + 0) + j, du, n, N);
        cdiff2(dfdv, F + n*(t*0 + 0) + j, dv, 1, N);
        jacob(c,dfdu,dfdv);
        A[n*0 + j] = sqrt(abs(c));
    }

    // j = n-1, k = 0
    fdiff2(dfdu, F + n*(t*0 + 0) + n-1, du, n, N);
    bdiff2(dfdv, F + n*(t*0 + 0) + n-1, dv, 1, N);
    jacob(c,dfdu,dfdv);
    A[n*0 + n-1] = sqrt(abs(c));

    for (k = 1; k < t-1; ++k) {
        // j = 0
        cdiff2(dfdu, F + n*(t*0 + k) + 0, du, n, N);
        fdiff2(dfdv, F + n*(t*0 + k) + 0, dv, 1, N);
        jacob(c,dfdu,dfdv);
        A[n*k + 0] = sqrt(abs(c));

        for (j = 1; j < n-1; ++j) {
            cdiff2(dfdu, F + n*(t*0 + k) + j, du, n, N);
            cdiff2(dfdv, F + n*(t*0 + k) + j, dv, 1, N);
            jacob(c,dfdu,dfdv);
            A[n*k + j] = sqrt(abs(c));
        }

        // j = n-1
        cdiff2(dfdu, F + n*(t*0 + k) + n-1, du, n, N);
        bdiff2(dfdv, F + n*(t*0 + k) + n-1, dv, 1, N);
        jacob(c,dfdu,dfdv);
        A[n*k + n-1] = sqrt(abs(c));
    }

    // j = 0, k = t-1
    bdiff2(dfdu, F + n*(t*0 + t-1) + 0, du, n, N);
    fdiff2(dfdv, F + n*(t*0 + t-1) + 0, dv, 1, N);
    jacob(c,dfdu,dfdv);
    A[n*(t-1) + 0] = sqrt(abs(c));

    // k = t-1
    for (j = 1; j < n-1; ++j) {
        bdiff2(dfdu, F + n*(t*0 + t-1) + j, du, n, N);
        cdiff2(dfdv, F + n*(t*0 + t-1) + j, dv, 1, N);
        jacob(c,dfdu,dfdv);
        A[n*(t-1) + j] = sqrt(abs(c));
    }

    // j = n-1, k = t-1
    bdiff2(dfdu, F + n*(t*0 + t-1) + n-1, du, n, N);
    bdiff2(dfdv, F + n*(t*0 + t-1) + n-1, dv, 1, N);
    jacob(c,dfdu,dfdv);
    A[n*(t-1) + n-1] = sqrt(abs(c));

    return;

}