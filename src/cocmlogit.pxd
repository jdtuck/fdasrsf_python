cdef extern from "ocmlogit_warp_grad.h":
    void ocmlogit_warp_grad(int *n1, int *T1, int *m1, double *alpha, double *nu, double *q, int *y, int *max_itri, double *toli, double *deltaOi, double *deltagi, int *displayi, double *gamout, double *Oout)
