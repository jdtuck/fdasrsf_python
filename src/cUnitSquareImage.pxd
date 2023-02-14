# cython: language_level=2
from libcpp cimport bool

cdef extern from "UnitSquareImage.h":
    void findgrad2D(double *dfdu, double *dfdv, double *f, int n, int t, int d);
    int check_crossing(double *f, int n, int t, int D);
