cdef extern from "DP.h":
    int DP(double *q1, double *q2, int n1, int N1, double lam1, int pen, int Disp, double *yy)
