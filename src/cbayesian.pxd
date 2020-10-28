# cython: language_level=2
from libcpp cimport bool
cimport cyarma
from cyarma cimport vec

cdef extern from "bayesian.h":
    vec calcY(double area, vec gy);
    vec cuL2norm2(vec x, vec y); 
    double trapzCpp(vec x, vec y); 
    double order_l2norm(vec x, vec y);
