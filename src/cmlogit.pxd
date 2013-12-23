cdef extern from "mlogit_warp_grad.h":
    void mlogit_warp_grad(int *m1, int *m2, double *alpha, double *beta, double *ti, double *gami, double *q, int *y, int *max_itri, double *toli, double *deltai, int *displayi, double *gamout)
