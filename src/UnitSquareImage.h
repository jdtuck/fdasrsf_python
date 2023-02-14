#ifndef UNITSQUAREIMAGE_H
#define UNITSQUAREIMAGE_H

#include <cstring>
#include <cmath>
#include <iostream>
#include "ImageRegister.h"

using namespace std;


void findgrad(double *dfdu, double *dfdv, const double *f, int n, int t);
void findgrad2D(double *dfdu, double *dfdv, double *f, int n, int t, int d);
void multfact_image(double *multfact, const double *dfdu, const double *dfdv, int n, int t, int d);
void surface_to_q(double *q, const double *f, const double *multfact, int n, int t, int d);
void Calculate_Distance(double *H, const double *q1, const double *q2, int n, int t, int d);
void findphistar(double *w, double *q, double *b, int n, int t, int d, int K);
void findupdategam(double *gamupdate, const double *v, const double *w, const double *b, int n, int t, int d, int K);
void updategam(double *gamnew, const double *gamupdate, const double *gamid, double eps, int n, int t, int D);
void Apply_gam_gamid(double *gamcum, const double *gamid, const double *gaminc, int m, int n);
void Apply_Gamma_Surf(double *Fnew, const double *F, const double *gam, int n, int d);
int check_crossing(double *f, int n, int t, int D);
int ReparamSurf(double *Fnew, double *gamnew, double *H,
        double *Ft, const double *Fm, const double *gam,
        double *b, const double *gamid,
        const int n, const int t, const int d, const int D, const int K,
        const double eps, const double tol, const int itermax);

void InvtGamma(double *gaminv, const double *gam, const double *gamid, int n);
void jacob_image(double *A, const double *F, int n, int t);

#endif