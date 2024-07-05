from libcpp cimport bool
cimport cyarma
from cyarma cimport vec

cdef extern from "rbfgs.h":
    vec rlbfgs_optim(vec q1, vec q2, vec time, int maxiter, double lam, int penalty);
