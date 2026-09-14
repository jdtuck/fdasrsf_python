#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dp_grid.h"
#include "dp_nbhd.h"

/* pen selects the warping penalty weighted by lam1: 0 = none, 1 = roughness,
 * 2 = l2gam, 3 = l2psi, 4 = geodesic (the DP_PEN_* constants in dp_grid.h).
 *
 * Returns 0 on success, or -1 if a work buffer could not be allocated, in
 * which case G, T and size are left untouched. */
int DynamicProgrammingQ2(double *Q1, double *T1, double *Q2, double *T2, int m1, int n1, int n2,
double *tv1, double *tv2, int n1v, int n2v, double *G, double *T, double *size, double lam1,
size_t nbhd_dim, int pen){
  int *idxv1 = 0;
  int *idxv2 = 0;
  double *E = 0; /* E[ntv1*j+i] = cost of best path to (tv1[i],tv2[j]) */
  int *P = 0; /* P[ntv1*j+i] = predecessor of (tv1[i],tv2[j]) along best path */
  size_t nbhd_count; /* Number of indexes */
  Pair *dp_nbhd = 0;
  int status = -1;

  /* Sizes are computed in size_t: n1v*n2v overflows int for large grids. */
  idxv1=(int*)malloc((size_t)n1v*sizeof(int));
  idxv2=(int*)malloc((size_t)n2v*sizeof(int));
  E=(double*)malloc((size_t)n1v*(size_t)n2v*sizeof(double));
  P=(int*)calloc((size_t)n1v*(size_t)n2v,sizeof(int));
  dp_nbhd = dp_generate_nbhd(nbhd_dim, &nbhd_count);

  if ( idxv1 && idxv2 && E && P && dp_nbhd )
  {
    /* dp_costs() needs indexes for gridpoints precomputed */
    dp_all_indexes( T1, n1, tv1, n1v, idxv1 );
    dp_all_indexes( T2, n2, tv2, n2v, idxv2 );

    /* Compute cost of best path from (0,0) to every other grid point */
    dp_costs( Q1, T1, n1, Q2, T2, n2,
      m1, tv1, idxv1, n1v, tv2, idxv2, n2v, E, P, lam1, pen,
      nbhd_count, dp_nbhd );

    /* Reconstruct best path from (0,0) to (1,1) */
    *size = dp_build_gamma( P, tv1, n1v, tv2, n2v, G, T );
    status = 0;
  }

  // free allocated memory
  free(dp_nbhd);
  free(idxv1); free(idxv2); free(E); free(P);

  return status;
}
