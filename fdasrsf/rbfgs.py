"""
Utility functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
from scipy.integrate import trapz

class rlbfgs:

    """
    This class provides alignment methods for functional data using the SRVF framework
    using the Riemannian limited memory BFGS solver.  The solver is designed to operate 
    on the positive orthant of the unit hypersphere in L^2([0,1],R). The set of all functions
    h=\sqrt{\dot{\gamma}}, where \gamma is a diffeomorphism, is that manifold.

    The inputs q1 and q2 are the square root velocity functions of curves in
    R^n to be aligned. Here, q2 will be aligned to q1.

    Usage:  obj = rlbfgs(q1,q2,t)

    :param q1: (M,N): matrix defining srvf of dimension M of N samples
    :param q2: (M,N): matrix defining srvf of dimension M of N samples
    :param t: time vector of length N

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  27-Oct-2020
    """

    def __init__(self, q1, q2, t):
        """
        Construct an instance of the rlbfgs class
        :param q1: (M,N): matrix defining srvf of dimension M of N samples
        :param q2: (M,N): matrix defining srvf of dimension M of N samples
        :param t: time vector of length N
        """

        self.q1 = q1
        self.q2 = q2
        self.t = t
        self.T = t.shape[0]
    
    def dist(self, f1, f2):
        d = np.real(np.arccos(self.inner(f1,f2)))
        return d
    
    def typicaldist(self):
        return np.pi/2
    
    def proj(self, f, v):
        out = v - f * trapz(f*v, self.t)
        return out
    
    def log(self, f1, f2):
        v = self.proj(f1, f2-f1)
        di = self.dist(f1, f2)
        if di > 1e-6:
            nv = self.norm(v)
            v = v * (di / nv)
        
        return v
    
    def exp(self, f1, v, delta=1):
        vd = delta*v
        nrm_vd = self.norm(vd)
        
        # Former versions of Manopt avoided the computation of sin(a)/a for
        # small a, but further investigations suggest this computation is
        # well-behaved numerically.
        if nrm_vd > 0:
            f2 = f1*np.cos(nrm_vd) + vd*(np.sin(nrm_vd)/nrm_vd)
        else:
            f2 = f1
        
        return f2
    
    def transp(self, f1, f2, v):
        """
        Isometric vector transport of d from the tangent space at x1 to x2.
        This is actually a parallel vector transport, see (5) in
        http://epubs.siam.org/doi/pdf/10.1137/16M1069298
        "A Riemannian Gradient Sampling Algorithm for Nonsmooth Optimization
        on Manifolds", by Hosseini and Uschmajew, SIOPT 2017
        """
        w = self.log(f1, f2)
        dist_f1f2 = self.norm(w)
        if dist_f1f2 > 0:
            u = w / dist_f1f2
            utv = self.inner(u,v)
            Tv = v + (np.cos(dist_f1f2)-1)*utv*u -  np.sin(dist_f1f2)*utv*f1
        else:
            Tv = v
        
        return Tv

    def inner(self, v1, v2):
        return trapz(v1*v2,self.t)
    
    def norm(self, v):
        return np.sqrt(trapz(v**2,self.t))
    
    def zerovec(self):
        return np.zeros(self.T)


    

