cdef extern from "fpls_warp_grad.h":
    void fpls_warp_grad(int *m1, int *n1, double *ti, double *gami, double *qf, double *qg, double *wf, double *wg,
                        int *max_itri, double *toli, double *deltai, int *displayi, double *gamout)
