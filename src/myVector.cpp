#include <iostream>
#include <cstring>
#include <cmath>
#include "myVector.h"

using namespace std;

double InProd(const double *u, const double *v, int d)
{
    double innp = 0.0;

    for (int i = 0; i < d; i++) {
        innp += u[i]*v[i];
    }

    return innp;
}

void GramSchmitd(double *x, int &n, int d)
{
    // n=# of basis; d= dimension for each basis element
    double innp;
    int cnt = 0, k;

    cnt = n; //initial # of vectors

    // 1st vector
    k = 0;
    innp = sqrt(InProd(x,x,d));
    for (int i=0; i<d; ++i) {
        x[d*k + i] /= innp;
    }

    // 2nd to the last
    k = 1;
    do{
        // kth vector
        for (int j=0; j<k; ++j) {
            innp = InProd(x+d*k, x+d*j,d);

            for (int i=0; i<d; ++i) {
                x[d*k+i] -= innp*x[d*j+i];
            }
        }

        innp = sqrt(InProd(x+d*k,x+d*k,d));
        if (innp>EPS) {
            for (int i=0; i<d; ++i) {
                x[d*k+i] /= innp;
            }
            k += 1;
        }
        else {
            for (int i=0; i<d; ++i) {
                x[d*k+i] = x[d*cnt+i];
            }
            cnt -= 1;
        }
    } while (k < (cnt-1));

    n = cnt;
}

double innerSquare(const double *u, const double *v, int n1, int n2, int d)
{
    int N = n1*n2;
    double innp = 0.0, du, dv;

    du = 1.0/(n1-1);
    dv = 1.0/(n2-1);

    for (int i = 0; i < N*d; i++) {
        innp += u[i]*v[i];
    }

    innp *= du*dv;
    return innp;
}

void GramSchmitdSquare(double *x, int &n, int n1, int n2, int d)
{
    // n=# of basis; d= dimension for each basis element
    double innp;
    int cnt = 0, k, N = n1*n2*d, i, j;

    cnt = n-1;

    // 1st vector
    k = 0;
    innp = sqrt(innerSquare(x,x,n1,n2,d));
    for (i=0; i<N; i++) {
        x[N*k + i] /= innp;
    }

    // 2nd to the last
    k = 1;
    do{
        // kth vector
        for (j=0; j<k; j++) {
            innp = innerSquare(x+N*k, x+N*j,n1,n2,d);

            for (i=0; i<N; i++) {
                x[N*k+i] -= innp*x[N*j + i];
            }
        }

        innp = sqrt(innerSquare(x+N*k,x+N*k,n1,n2,d));
        if (innp>EPS) {
            for (i=0; i<N; i++) {
                x[N*k + i] /= innp;
            }
            k += 1;
        }
        else {
            for (i=0; i<N; i++) {
                x[N*k + i] = x[N*cnt + i];
            }
            cnt -= 1;
        }
    } while (k <= cnt);

    n = k;
}
