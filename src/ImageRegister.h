#ifndef SURFACECHRISTOFFEL_H
#define SURFACECHRISTOFFEL_H

#include <cmath>
#include <cstring>
using namespace std;

#define SQRT5 2.23606797749978969641

// Prototypes: -----------------------------------------------------------------
template <typename T> void cpyArray(T *x, const T *y, int n);

template <typename T> T innerProd(const T *x, const T *y, int N, int d);

template <typename T> T det2(const T *A);

template <typename T> T det3(const T *A);

template <typename T> inline void jacob(T z, const T x[2], const T y[2]);

template <typename T> void jacobMat(T *z, const T *x, const T *y, int n, int t);

template <typename T> void cross(T z[3], const T x[3], const T y[3]);

template <typename T> void cross2(T z, const T x[2], const T y[2]);

template <typename T> void matmul3(T *C, const T *A, const T *B);

template <typename T> void matvec3(T y[3], const T *A, const T x[3]);

template <typename T> void aat3(T *AA, const T *A);

template <typename T> void ata3(T *AA, const T *A);

template <typename T> void div3(T x[3], T div);

template <typename T> void mult3(T x[3], T mult);

template <typename T> void assign3(T *y, const T x[3], int s = 1);

template <typename T> T L2norm3(T x[3]);

template <typename T> T L2norm(T x, int s);

template <typename T> T cdiff(const T *F, T dx, int s);

template <typename T> T fdiff(const T *F, T dx, int s);

template <typename T> T bdiff(const T *F, T dx, int s);

template <typename T> void cdiff2(T df[2], const T *F, T dx, int s, int S);

template <typename T> void fdiff2(T df[2], const T *F, T dx, int s, int S);

template <typename T> void bdiff2(T df[2], const T *F, T dx, int s, int S);

template <typename T> void cdiff3(T df[3], const T *F, T dx, int s, int S);

template <typename T> void fdiff3(T df[3], const T *F, T dx, int s, int S);

template <typename T> void bdiff3(T df[3], const T *F, T dx, int s, int S);

template <typename T> void sort3(T x[3]);

template <typename T> void solvecubic(T c[3]);

template <typename T> void ldu3(T *A, int P[3]);

template <typename T>
void ldubsolve3(T x[3], const T y[3], const T *LDU, const int P[3]);

template <typename T> void unit3(T x[3]);

template <typename T> void svd3(T *U, T S[3], T *V, const T *A);

template <typename T> void thomas(T *x, const T *a, const T *b, T *c, int n);

template <typename T> void spline(T *D, const T *y, int n);

template <typename T> void lookupspline(T &t, int &k, T dist, T len, int n);

template <typename T> T evalspline(T t, const T D[2], const T y[2]);

template <typename T> void interp2(T *Dy, const T *Z, int nx, int ny);

template <typename T>
void evalinterp2(T u, T *Dx, T *zx, const T *Dy, const T *Z, int nx, int ny);

// Subroutines ============================================================
template <typename T> void cpyArray(T *x, const T *y, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = y[i];
  }
}

template <typename T> inline T innerProd(const T *x, const T *y, int N, int d) {
  T s = 0;
  for (int i = 0; i < d; i++) {
    s += x[N * i] * y[N * i];
  }
  return s;
}

template <typename T> inline void jacob(T z, const T x[2], const T y[2]) {
  z = x[1] * y[2] - x[2] * y[1];
}

template <typename T>
inline void jacobMat(T *z, const T *x, const T *y, int n, int t) {
  int i, N = n * t, d = 2;
  if (d == 2) {
    for (i = 0; i < N; ++i) {
      z[i] = x[N * 0 + i] * y[N * 1 + i] - x[N * 1 + i] * y[N * 0 + i];
    }
  }
}

template <typename T> inline void cross(T z[3], const T x[3], const T y[3]) {
  z[0] = x[1] * y[2] - x[2] * y[1];
  z[1] = -(x[0] * y[2] - x[2] * y[0]);
  z[2] = x[0] * y[1] - x[1] * y[0];
}

template <typename T> inline void cross2(T z, const T x[2], const T y[2]) {
  z = x[1] * y[2] - x[2] * y[1];
}

template <typename T> inline T cdiff(const T *F, T dx, int s) {
  return (F[s] - F[-s]) / (2 * dx);
}

template <typename T> inline T fdiff(const T *F, T dx, int s) {
  return (F[s] - F[0]) / dx;
}

template <typename T> inline T bdiff(const T *F, T dx, int s) {
  return (F[0] - F[-s]) / dx;
}

template <typename T>
inline void cdiff2(T df[2], const T *F, T dx, int s, int S) {
  int i;
  for (i = 0; i < 2; ++i) {
    df[i] = (F[i * S + s] - F[i * S - s]) / (2 * dx);
  }
}

template <typename T>
inline void fdiff2(T df[2], const T *F, T dx, int s, int S) {
  int i;
  for (i = 0; i < 2; ++i) {
    df[i] = (F[i * S + s] - F[i * S]) / (2 * dx);
  }
}

template <typename T>
inline void bdiff2(T df[2], const T *F, T dx, int s, int S) {
  int i;
  for (i = 0; i < 2; ++i) {
    df[i] = (F[i * S] - F[i * S - s]) / (2 * dx);
  }
}

template <typename T>
inline void cdiff3(T df[3], const T *F, T dx, int s, int S) {
  df[0] = (F[s] - F[-s]) / (2 * dx);
  df[1] = (F[S + s] - F[S - s]) / (2 * dx);
  df[2] = (F[2 * S + s] - F[2 * S - s]) / (2 * dx);
}

template <typename T>
inline void fdiff3(T df[3], const T *F, T dx, int s, int S) {
  df[0] = (F[s] - F[0]) / dx;
  df[1] = (F[S + s] - F[S]) / dx;
  df[2] = (F[2 * S + s] - F[2 * S]) / dx;
}

template <typename T>
inline void bdiff3(T df[3], const T *F, T dx, int s, int S) {
  df[0] = (F[0] - F[-s]) / dx;
  df[1] = (F[S] - F[S - s]) / dx;
  df[2] = (F[2 * S] - F[2 * S - s]) / dx;
}

template <typename T> inline void matmul3(T *C, const T *A, const T *B) {
  C[3 * 0 + 0] = A[3 * 0 + 0] * B[3 * 0 + 0] + A[3 * 1 + 0] * B[3 * 0 + 1] +
                 A[3 * 2 + 0] * B[3 * 0 + 2];
  C[3 * 1 + 0] = A[3 * 0 + 0] * B[3 * 1 + 0] + A[3 * 1 + 0] * B[3 * 1 + 1] +
                 A[3 * 2 + 0] * B[3 * 1 + 2];
  C[3 * 2 + 0] = A[3 * 0 + 0] * B[3 * 2 + 0] + A[3 * 1 + 0] * B[3 * 2 + 1] +
                 A[3 * 2 + 0] * B[3 * 2 + 2];

  C[3 * 0 + 1] = A[3 * 0 + 1] * B[3 * 0 + 0] + A[3 * 1 + 1] * B[3 * 0 + 1] +
                 A[3 * 2 + 1] * B[3 * 0 + 2];
  C[3 * 1 + 1] = A[3 * 0 + 1] * B[3 * 1 + 0] + A[3 * 1 + 1] * B[3 * 1 + 1] +
                 A[3 * 2 + 1] * B[3 * 1 + 2];
  C[3 * 2 + 1] = A[3 * 0 + 1] * B[3 * 2 + 0] + A[3 * 1 + 1] * B[3 * 2 + 1] +
                 A[3 * 2 + 1] * B[3 * 2 + 2];

  C[3 * 0 + 2] = A[3 * 0 + 2] * B[3 * 0 + 0] + A[3 * 1 + 2] * B[3 * 0 + 1] +
                 A[3 * 2 + 2] * B[3 * 0 + 2];
  C[3 * 1 + 2] = A[3 * 0 + 2] * B[3 * 1 + 0] + A[3 * 1 + 2] * B[3 * 1 + 1] +
                 A[3 * 2 + 2] * B[3 * 1 + 2];
  C[3 * 2 + 2] = A[3 * 0 + 2] * B[3 * 2 + 0] + A[3 * 1 + 2] * B[3 * 2 + 1] +
                 A[3 * 2 + 2] * B[3 * 2 + 2];
}

template <typename T> inline void matvec3(T y[3], const T *A, const T x[3]) {
  y[0] = A[3 * 0 + 0] * x[0] + A[3 * 1 + 0] * x[1] + A[3 * 2 + 0] * x[2];
  y[1] = A[3 * 0 + 1] * x[0] + A[3 * 1 + 1] * x[1] + A[3 * 2 + 1] * x[2];
  y[2] = A[3 * 0 + 2] * x[0] + A[3 * 1 + 2] * x[1] + A[3 * 2 + 2] * x[2];
}

template <typename T> inline void aat3(T *AA, const T *A) {
  AA[3 * 0 + 0] = A[3 * 0 + 0] * A[3 * 0 + 0] + A[3 * 1 + 0] * A[3 * 1 + 0] +
                  A[3 * 2 + 0] * A[3 * 2 + 0];
  AA[3 * 1 + 0] = A[3 * 0 + 0] * A[3 * 0 + 1] + A[3 * 1 + 0] * A[3 * 1 + 1] +
                  A[3 * 2 + 0] * A[3 * 2 + 1];
  AA[3 * 2 + 0] = A[3 * 0 + 0] * A[3 * 0 + 2] + A[3 * 1 + 0] * A[3 * 1 + 2] +
                  A[3 * 2 + 0] * A[3 * 2 + 2];

  AA[3 * 0 + 1] = AA[3 * 1 + 0];
  AA[3 * 1 + 1] = A[3 * 0 + 1] * A[3 * 0 + 1] + A[3 * 1 + 1] * A[3 * 1 + 1] +
                  A[3 * 2 + 1] * A[3 * 2 + 1];
  AA[3 * 2 + 1] = A[3 * 0 + 1] * A[3 * 0 + 2] + A[3 * 1 + 1] * A[3 * 1 + 2] +
                  A[3 * 2 + 1] * A[3 * 2 + 2];

  AA[3 * 0 + 2] = AA[3 * 2 + 0];
  AA[3 * 1 + 2] = AA[3 * 2 + 1];
  AA[3 * 2 + 2] = A[3 * 0 + 2] * A[3 * 0 + 2] + A[3 * 1 + 2] * A[3 * 1 + 2] +
                  A[3 * 2 + 2] * A[3 * 2 + 2];
}

template <typename T> inline void ata3(T *AA, const T *A) {
  AA[3 * 0 + 0] = A[3 * 0 + 0] * A[3 * 0 + 0] + A[3 * 0 + 1] * A[3 * 0 + 1] +
                  A[3 * 0 + 2] * A[3 * 0 + 2];
  AA[3 * 1 + 0] = A[3 * 0 + 0] * A[3 * 1 + 0] + A[3 * 0 + 1] * A[3 * 1 + 1] +
                  A[3 * 0 + 2] * A[3 * 1 + 2];
  AA[3 * 2 + 0] = A[3 * 0 + 0] * A[3 * 2 + 0] + A[3 * 0 + 1] * A[3 * 2 + 1] +
                  A[3 * 0 + 2] * A[3 * 2 + 2];

  AA[3 * 0 + 1] = AA[3 * 1 + 0];
  AA[3 * 1 + 1] = A[3 * 1 + 0] * A[3 * 1 + 0] + A[3 * 1 + 1] * A[3 * 1 + 1] +
                  A[3 * 1 + 2] * A[3 * 1 + 2];
  AA[3 * 2 + 1] = A[3 * 1 + 0] * A[3 * 2 + 0] + A[3 * 1 + 1] * A[3 * 2 + 1] +
                  A[3 * 1 + 2] * A[3 * 2 + 2];

  AA[3 * 0 + 2] = AA[3 * 2 + 0];
  AA[3 * 1 + 2] = AA[3 * 2 + 1];
  AA[3 * 2 + 2] = A[3 * 2 + 0] * A[3 * 2 + 0] + A[3 * 2 + 1] * A[3 * 2 + 1] +
                  A[3 * 2 + 2] * A[3 * 2 + 2];
}

template <typename T> inline void div3(T x[3], T div) {
  x[0] /= div;
  x[1] /= div;
  x[2] /= div;
}

template <typename T> inline void mult3(T x[3], T mult) {
  x[0] *= mult;
  x[1] *= mult;
  x[2] *= mult;
}

template <typename T> inline T det2(const T *A) {
  return A[2 * 0 + 0] * A[2 * 1 + 1] - A[2 * 0 + 1] * A[2 * 1 + 0];
}

template <typename T> inline T det3(const T *A) {
  return A[3 * 0 + 0] *
             (A[3 * 1 + 1] * A[3 * 2 + 2] - A[3 * 2 + 1] * A[3 * 1 + 2]) -
         A[3 * 1 + 0] *
             (A[3 * 0 + 1] * A[3 * 2 + 2] - A[3 * 2 + 1] * A[3 * 0 + 2]) +
         A[3 * 2 + 0] *
             (A[3 * 0 + 1] * A[3 * 1 + 2] - A[3 * 1 + 1] * A[3 * 0 + 2]);
}

template <typename T> inline void assign3(T *y, const T x[3], int s) {
  y[0] = x[0];
  y[s] = x[1];
  y[2 * s] = x[2];
}

template <typename T> inline T L2norm3(T x[3]) {
  return (T)sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

template <typename T> inline T L2norm(T x, int s) {
  double temp = 0;
  for (int i = 0; i < s; ++i) {
    temp += x[i] * x[i];
  }
  return (T)sqrt(temp);
}

template <typename T> inline void sort3(T x[3]) {
  T tmp;
  int k;

  k = (x[1] <= x[0] || x[2] <= x[0]) + (x[2] <= x[0] && x[2] <= x[1]);
  tmp = x[2];
  x[2] = x[k];
  x[k] = tmp;

  k = (x[1] <= x[0]);
  tmp = x[1];
  x[1] = x[k];
  x[k] = tmp;
}

template <typename T> void solvecubic(T c[3]) {
  const double sq3d2 = 0.86602540378443864676, c2d3 = c[2] / 3.0,
               c2sq = ((double)c[2]) * c[2], Q = (3.0 * c[1] - c2sq) / 9.0,
               R = (c[2] * (9.0 * c[1] - 2.0 * c2sq) - 27.0 * c[0]) / 54.0;
  double tmp, t, sint, cost;

  if (Q < 0) {
    tmp = 2 * sqrt(-Q);
    t = acos(R / sqrt(-Q * Q * Q)) / 3;
    cost = tmp * cos(t);
    sint = tmp * sin(t);

    c[0] = (T)(cost - c2d3);

    cost = -0.5 * cost - c2d3;
    sint = sq3d2 * sint;

    c[1] = (T)(cost - sint);
    c[2] = (T)(cost + sint);
  } else {
#if __STDC_VERSION__ >= 199901L
    tmp = cbrt(R);
#else
    tmp = pow(R, 1.0 / 3);
#endif
    c[0] = (T)(-c2d3 + 2 * tmp);
    c[1] = c[2] = (T)(-c2d3 - tmp);
  }
}

template <typename T> void ldu3(T *A, int P[3]) {
  int tmp, k;
  P[1] = 1;
  P[2] = 2;

  P[0] = ((fabs(A[3 * 0 + 0]) < fabs(A[3 * 1 + 0])) |
          (fabs(A[3 * 0 + 0]) < fabs(A[3 * 2 + 0]))) +
         ((fabs(A[3 * 0 + 0]) < fabs(A[3 * 2 + 0])) &
          (fabs(A[3 * 1 + 0]) < fabs(A[3 * 2 + 0])));
  P[P[0]] = 0;

  k = 1 + (fabs(A[3 * P[1] + 1]) < fabs(A[3 * P[2] + 1]));
  tmp = P[1];
  P[1] = P[k];
  P[k] = tmp;

  if (A[3 * P[0] + 0] != 0) {
    A[3 * P[1] + 0] = A[3 * P[1] + 0] / A[3 * P[0] + 0];
    A[3 * P[2] + 0] = A[3 * P[2] + 0] / A[3 * P[0] + 0];
    A[3 * P[0] + 1] = A[3 * P[0] + 1] / A[3 * P[0] + 0];
    A[3 * P[0] + 2] = A[3 * P[0] + 2] / A[3 * P[0] + 0];
  }

  A[3 * P[1] + 1] =
      A[3 * P[1] + 1] - A[3 * P[0] + 1] * A[3 * P[1] + 0] * A[3 * P[0] + 0];

  if (A[3 * P[1] + 1] != 0) {
    A[3 * P[2] + 1] = (A[3 * P[2] + 1] -
                       A[3 * P[0] + 1] * A[3 * P[2] + 0] * A[3 * P[0] + 0]) /
                      A[3 * P[1] + 1];
    A[3 * P[1] + 2] = (A[3 * P[1] + 2] -
                       A[3 * P[0] + 2] * A[3 * P[1] + 0] * A[3 * P[0] + 0]) /
                      A[3 * P[1] + 1];
  }

  A[3 * P[2] + 2] = A[3 * P[2] + 2] -
                    A[3 * P[0] + 2] * A[3 * P[2] + 0] * A[3 * P[0] + 0] -
                    A[3 * P[1] + 2] * A[3 * P[2] + 1] * A[3 * P[1] + 1];
}

template <typename T>
inline void ldubsolve3(T x[3], const T y[3], const T *LDU, const int P[3]) {
  x[P[2]] = y[2];
  x[P[1]] = y[1] - LDU[3 * P[2] + 1] * x[P[2]];
  x[P[0]] = y[0] - LDU[3 * P[2] + 0] * x[P[2]] - LDU[3 * P[1] + 0] * x[P[1]];
}

template <typename T> inline void unit3(T x[3]) {
  double tmp = sqrt(((double)x[0]) * x[0] + ((double)x[1]) * x[1] +
                    ((double)x[2]) * x[2]);
  x[0] = (T)(x[0] / tmp);
  x[1] = (T)(x[1] / tmp);
  x[2] = (T)(x[2] / tmp);
}

template <typename T> void svd3(T *U, T S[3], T *V, const T *A) {
  const T thr = (T)(1e-10);
  int P[3], k;
  T y[3], AA[3][3], LDU[3][3];

  ata3((double *)AA, A);

  S[2] = -AA[0][0] - AA[1][1] - AA[2][2];
  S[1] = AA[0][0] * AA[1][1] + AA[2][2] * AA[0][0] + AA[2][2] * AA[1][1] -
         AA[2][1] * AA[1][2] - AA[2][0] * AA[0][2] - AA[1][0] * AA[0][1];
  S[0] = AA[2][1] * AA[1][2] * AA[0][0] + AA[2][0] * AA[0][2] * AA[1][1] +
         AA[1][0] * AA[0][1] * AA[2][2] - AA[0][0] * AA[1][1] * AA[2][2] -
         AA[1][0] * AA[2][1] * AA[0][2] - AA[2][0] * AA[0][1] * AA[1][2];

  solvecubic(S);

  if (S[0] < 0)
    S[0] = 0;
  if (S[1] < 0)
    S[1] = 0;
  if (S[2] < 0)
    S[2] = 0;

  sort3(S);

  memcpy(LDU, AA, sizeof(LDU));
  LDU[0][0] -= S[0];
  LDU[1][1] -= S[0];
  LDU[2][2] -= S[0];

  ldu3((T *)LDU, P);

  y[0] = y[1] = y[2] = 0;
  k = (fabs(LDU[P[1]][1]) < fabs(LDU[P[0]][0]) ||
       fabs(LDU[P[2]][2]) < fabs(LDU[P[0]][0])) +
      (fabs(LDU[P[2]][2]) < fabs(LDU[P[0]][0]) &&
       fabs(LDU[P[2]][2]) < fabs(LDU[P[1]][1]));
  y[k] = 1;

  ldubsolve3(V + (3 * 0 + 0), y, (T *)LDU, P);

  memcpy(LDU, AA, sizeof(LDU));
  LDU[0][0] -= S[2];
  LDU[1][1] -= S[2];
  LDU[2][2] -= S[2];

  ldu3((T *)LDU, P);

  y[0] = y[1] = y[2] = 0;
  k = (fabs(LDU[P[1]][1]) <= fabs(LDU[P[0]][0]) ||
       fabs(LDU[P[2]][2]) <= fabs(LDU[P[0]][0])) +
      (fabs(LDU[P[2]][2]) <= fabs(LDU[P[0]][0]) &&
       fabs(LDU[P[2]][2]) <= fabs(LDU[P[1]][1]));
  y[k] = 1;

  ldubsolve3(V + (3 * 2 + 0), y, (T *)LDU, P);

  cross(V + (3 * 1 + 0), V + (3 * 2 + 0), V + (3 * 0 + 0));

  k = (S[0] > thr) + (S[1] > thr) + (S[2] > thr);

  switch (k) {
  case 0:
    memcpy(U, V, 9 * sizeof(T));
    break;
  case 1:
    matvec3(U + (3 * 0 + 0), A, V + (3 * 0 + 0));

    y[0] = y[1] = y[2] = 0;
    k = (fabs(U[3 * 0 + 1]) <= fabs(U[3 * 0 + 0]) ||
         fabs(U[3 * 0 + 2]) <= fabs(U[3 * 0 + 0])) +
        (fabs(U[3 * 0 + 2]) <= fabs(U[3 * 0 + 0]) &&
         fabs(U[3 * 0 + 2]) <= fabs(U[3 * 0 + 1]));
    y[k] = 1;

    cross(U + (3 * 1 + 0), y, U + (3 * 0 + 0));
    cross(U + (3 * 2 + 0), U + (3 * 0 + 0), U + (3 * 1 + 0));
    break;
  case 2:
    matvec3(U + (3 * 0 + 0), A, V + (3 * 0 + 0));
    matvec3(U + (3 * 1 + 0), A, V + (3 * 1 + 0));

    cross(U + (3 * 2 + 0), U + (3 * 0 + 0), U + (3 * 1 + 0));
    break;
  case 3:
    matmul3(U, A, V);
    break;
  }

  unit3(V + (3 * 0 + 0));
  unit3(V + (3 * 1 + 0));
  unit3(V + (3 * 2 + 0));

  unit3(U + (3 * 0 + 0));
  unit3(U + (3 * 1 + 0));
  unit3(U + (3 * 2 + 0));

  S[0] = (T)sqrt(S[0]);
  S[1] = (T)sqrt(S[1]);
  S[2] = (T)sqrt(S[2]);
}

template <typename T> void thomas(T *x, const T *a, const T *b, T *c, int n) {
  T tmp;
  int i;

  c[0] /= b[0];
  x[0] /= b[0];

  for (i = 1; i < n; ++i) {
    tmp = 1 / (b[i] - c[i - 1] * a[i]);
    c[i] *= tmp;
    x[i] = (x[i] - x[i - 1] * a[i]) * tmp;
  }

  for (i = n - 2; i >= 0; --i) {
    x[i] -= c[i] * x[i + 1];
  }
}

template <typename T> void spline(T *D, const T *y, int n) {
  int i;
  T *a, *b, *c;

  a = new T[n];
  b = new T[n];
  c = new T[n];

  if (n < 4) {
    a[0] = 0;
    b[0] = 2;
    c[0] = 1;
    D[0] = 3 * (y[1] - y[0]);

    a[n - 1] = 1;
    b[n - 1] = 2;
    c[n - 1] = 0;
    D[n - 1] = 3 * (y[n - 1] - y[n - 2]);
  } else {
    a[0] = 0;
    b[0] = 2;
    c[0] = 4;
    D[0] = -5 * y[0] + 4 * y[1] + y[2];

    a[n - 1] = 4;
    b[n - 1] = 2;
    c[n - 1] = 0;
    D[n - 1] = 5 * y[n - 1] - 4 * y[n - 2] - y[n - 3];
  }

  for (i = 1; i < n - 1; ++i) {
    a[i] = 1;
    b[i] = 4;
    c[i] = 1;
    D[i] = 3 * (y[i + 1] - y[i - 1]);
  }

  thomas(D, a, b, c, n);

  delete[] a;
  delete[] b;
  delete[] c;
}

template <typename T>
inline void lookupspline(T &t, int &k, T dist, T len, int n) {
  t = (n - 1) * dist / len;
  k = (int)floor(t);

  k = (k > 0) * k;
  k += (k > n - 2) * (n - 2 - k);

  t -= k;
}

template <typename T> inline T evalspline(T t, const T D[2], const T y[2]) {
  const T c[4] = {y[0], D[0], 3 * (y[1] - y[0]) - 2 * D[0] - D[1],
                  2 * (y[0] - y[1]) + D[0] + D[1]};

  return t * (t * (t * c[3] + c[2]) + c[1]) + c[0];
}

template <typename T> void interp2(T *Dy, const T *Z, int nx, int ny) {

  for (int i = 0; i < ny; ++i) {
    spline(Dy + (nx * i + 0), Z + (nx * i + 0), nx);
  }
}

template <typename T>
void evalinterp2(T u, T *Dx, T *zx, const T *Dy, const T *Z, int nx, int ny) {

  for (int i = 0; i < ny; ++i) {
    zx[i] = evalspline(u, Dy + (nx * i + 0), Z + (nx * i + 0));
  }

  spline(Dx, zx, ny);
}

#endif // !SURFACE_H
