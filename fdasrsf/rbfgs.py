"""
Utility functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import time
from scipy.integrate import trapz, cumtrapz

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
    :param q2Opt: optimally aligned srvf
    :param gammaOpt: optimal warping function
    :param cost: final cost
    :param info: dictionary consisting of info about the iterations
    
    Wen Huang, Kyle A. Gallivan, Anuj Srivastava, Pierre-Antoine Absil. "Riemannian
    Optimization for Elastic Shape Analysis", Short version, The 21st International
    Symposium on Mathematical Theory of Networks and Systems (MTNS 2014).

    Code based on rlbfgs.m in Manopt: www.manopt.org

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
    
    def solve(self, maxiter=30, verb=0):
        """
        Run solver

        :param maxiter: maximum number of interations
        :param verb: integer used to tune the amount of output
        """
        
        # @todo add options to parameters if needed
        # terminates if the norm of the gradient drops below this
        tolgradnorm = 1e-3
        # terminates if more than seconds elapsed
        maxtime = np.inf
        # minimum norm of tangent vector that points from current to next
        minstepsize = 1e-10
        # number of previous iterations the program remembers
        memory = 30
        memory = max(memory,0)
        # the cautious step needs a real function that has value 0 at t=0
        strict_inc_func = lambda t : 1e-4*t
        ls_max_steps = 25

        options = {"tolgradnorm": tolgradnorm, "maxtime": maxtime, "memory":memory, 
                   "strict_inc_func":strict_inc_func, "ls_max_steps":ls_max_steps,
                   "maxiter":maxiter, "minstepsize":minstepsize}


        timetic = time.time()
        ## Initialization of Variables
        htilde = np.ones(self.T)
        q2tilde = self.q2

        # number of iterations since last restart
        j = 0

        # Total number of BFGS iterations
        k = 0

        # list to store step vectors which point from h_id to h_{k+1}
        # for k indexing the last iterations, capped at option memory
        sHistory = [None] * memory

        # list store differences for latest k's for the gradient at time
        # k+1 and the gradient at time k
        yHistory = [None] * memory

        # stores the reciprocal of the inner product between 
        # sHistory[k] and yHistory[k]
        rhoHistory = [None] * memory

        # scaling of direction given by getDirection
        alpha = 1

        # scaling of initial matrix, Barzilai-Borwein
        scaleFactor = 1

        # Norm of the step
        stepsize = 1

        # sores wether the step is accepted byt he cautious update check
        accpeted = True

        # compute cost function and its gradient
        hCurCost, hCurGradient = self.alignment_costgrad(q2tilde)
        hCurGradNorm = self.norm(hCurGradient)

        # line-search statistics for recording in info
        lsstats = {"costevals":0,"stepsize":0.0,"alpha":0.0}

        # flag to control restarting scheme to avoid infinite loops
        ultimatum = False

        if verb >= 2:
            print(' iter                   cost val            grad. norm           alpha\n')
        
        # stats
        info = []
        stats = {"iter":k, "cost":hCurCost, "gradnorm":hCurGradNorm,"stepsize":np.nan,
                "time":time.time() - timetic, "accepted":None, "linesearch": lsstats}
        info.append(stats)

        while True:
            if verb >= 2:
                print('%5d    %+.16e        %.8e      %.4e\n' % (k, hCurCost, hCurGradNorm, alpha))
            
            #Start timing this iteration
            timetic = time.time()

            # run standard stopping criterion checks
            stop = self.stoppingcriterion(options, info, k+1)

            if stop == 0:
                if stats["stepsize"] < options["minstepsize"]:
                    if not ultimatum:
                        if verb >= 2:
                            print('stepsize is too small, restarting the bfgs procedure at the current point.\n')
                        j = 0
                        ultimatum = True
                    else:
                        stop = 1
                else:
                    # we are not in trouble: list the ultimatum if it was on
                    ultimatum = False
            
            if stop > 0:
                break
            
            # compute BFGS direction
            p = self.getDirection(hCurGradient, sHistory, yHistory, rhoHistory,
                                  scaleFactor, min(j,memory))

            # execute line-search
            in_prod = self.inner(hCurGradient,p)
            stepsize, hNext, lsstats = self.linesearch_hint(p, hCurCost, in_prod, q2tilde, options)

            # Iterative update of optimal diffeomorphism and q2 via group action
            htilde = self.group_action_SRVF(htilde,hNext)
            q2tilde = self.group_action_SRVF(q2tilde,hNext)

            # Record the BFGS step-multiplier alpha which was effectively
            # selected. Toward convergence, we hope to see alpha = 1.
            alpha = stepsize/self.norm(p)
            step = alpha*p

            # query cost and gradient at the candidate new point
            hNextCost, hNextGradient = self.alignment_costgrad(q2tilde)

            # compute sk and yk
            sk = step
            yk = hNextGradient-hCurGradient

            # Computation of the BFGS step is invariant under scaling of sk and
            # yk by a common factor. For numerical reasons, we scale sk and yk
            # so that sk is a unit norm vector.
            norm_sk = self.norm(sk)
            sk = sk/norm_sk
            yk = yk/norm_sk

            inner_sk_yk = self.inner(sk,yk)
            inner_sk_sk = self.norm(sk)**2   # ensures nonnegativity

            # If the cautious step is accepted (which is the intended
            # behavior), we record sk, yk, and rhok and need to do some
            # housekeeping. If the cautious step is rejected, these are not
            # recorded. In all cases, hNext is the next iterate: the notion of
            # accept/reject here is limited to whether or not we keep track of
            # sk, yk, rhok to update the BFGS operator.
            cap = options["strict_inc_func"](hCurGradNorm)
            if inner_sk_sk != 0 and (inner_sk_yk/inner_sk_sk) >= cap:
                accepted = True

                rhok = 1/inner_sk_yk

                scaleFactor = inner_sk_yk/self.norm(yk)**2

                # Time to store the vectors sk, yk and the scalar rhok

                # If we are out of memory
                if j>=memory:
                    # sk and yk are saved from 1 to the end with the most
                    # current recorded to the rightmost hand side of the cells
                    # that are occupied. When memory is full, do a shift so
                    # that the rightmost is earliest and replace it with the
                    # most recent sk, yk.
                    if memory > 1:
                        tmp = sHistory[1:]
                        tmp.append(sHistory[0])
                        sHistory = tmp
                        tmp = yHistory[1:]
                        tmp.append(yHistory[0])
                        yHistory = tmp
                        tmp = rhoHistory[1:]
                        tmp.append(rhoHistory[0])
                        rhoHistory = tmp
                    if memory > 0:
                        sHistory[memory] = sk
                        yHistory[memory] = yk
                        rhoHistory[memory] = rhok
                    # if we are not out of memory
                else:
                    sHistory[j+1] = sk
                    yHistory[j+1] = yk
                    rhoHistory[j+1] = rhok
                
                j += 1

                # the cautious step is rejected we do not store sk, yk, and rhok
            else:
                accepted = False
            
            # update variables to new iterate
            hCurGradient = hNextGradient
            hCurGradNorm = self.norm(hNextGradient)
            hCurCost = hNextCost

            # iter is the number of iterations we have accomplished.
            k += 1
            stats = {"iter":k, "cost":hCurCost, "gradnorm":hCurGradNorm,"stepsize":np.nan,
                     "time":time.time() - timetic, "accepted":accepted, "linesearch": lsstats}
            info.append(stats)
        
        self.info = info[0:(k+1)]
        self.gammaOpt = np.zeros(self.T)
        self.gammaOpt[1:] = cumtrapz(htilde**2,self.t)
        self.q2Opt = q2tilde
        self.cost = hCurCost

        if verb >= 1:
            print('Total time is %f [s] (excludes statsfun)\n' % info[-1].time)
        
        return

    def alignment_cost(self, h, q2k):
        """
        Evaluate the cost function f = ||q1 - ((q2,hk),h)||^2.
        h=sqrt{\dot{\gamma}} is a sequential update of cumulative warping hk
        """
        q2new = self.group_action_SRVF(q2k,h)
        f = self.normL2(self.q1-q2new)**2

        return f

    def alignment_costgrad(self, q2k):
        """
        Evaluate the cost function f = ||q1 - (q2,hk)||^2, and
        evaluate the gradient g = grad f in the tangent space of identity.
        hk=sqrt{\dot{\gamma_k}} is the cumulative warping of q2 produced by an
        iterative sequential optimization algorithm.
        """
        t = self.t
        T = self.T
        q1 = self.q1

        # compute cost
        f = self.normL2(q1-q2k)**2

        # compute cost gradient
        q2kdot = np.gradient(q2k, 1/(T-1))
        dq = q1-q2k
        v = np.zeros(T)
        tmp = dq*q2kdot
        tmp1 = dq*q2k
        v[1:] = 2*cumtrapz(tmp.sum(axis=0)-tmp1.sum(axis=0), t)
        g = v - trapz(v,t)

        return f, g 
    
    def getDirection(self, hCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, j):
        """
        BFGS step, see Wen's paper for details. This function takes in a tangent
        vector g, and applies an approximate inverse Hessian P to it to get Pg.
        Then, -Pg is returned. Parallel transport is not needed for this problem
        since we always work in the tangent space of identity.
        """
        q = hCurGradient
        inner_s_q = np.zeros(j)

        for i in range(j,0,-1):
            inner_s_q[i] = rhoHistory[i] * self.inner(sHistory[i],q)
            q = q - inner_s_q[i] * yHistory[i]
        
        r = scaleFactor * q

        for i in range(0,j):
            omega = rhoHistory[i] * self.inner(yHistory[i],r)
            r = r + (inner_s_q[i]-omega)*sHistory[i]
    
        direction = -r 

        return direction
    
    def linesearch_hint(self, d, f0, df0, q2k, options):
        """
        Armijo line-search based on the line-search hint in the problem structure.
        
        Base line-search algorithm for descent methods, based on a simple
        backtracking method. The search direction provided has to be a descent
        direction, as indicated by a negative df0 = directional derivative of f
        at the identity element along d.
        
        The algorithm selects a hardcoded initial step size. If that
        step does not fulfill the Armijo sufficient decrease criterion, that step
        size is reduced geometrically until a satisfactory step size is obtained
        or until a failure criterion triggers.
        
        Below, the step is constructed as alpha*d, and the step size is the norm
        of that vector, thus: stepsize = alpha*norm_d. The step is executed by
        computing the exponential mapping exp_{hid}(alpha*d), giving newh.
        """
        q1 = self.q1
        contraction_factor = .5
        suff_decr = 1e-6
        max_ls_steps = 25
        ls_backtrack = True
        ls_force_decrease = True

        # init alpha
        alpha = 1

        # Identity element
        hid = np.ones(self.T)

        # Make the chosen step and compute cost there
        newh = self.exp(hid, d, alpha)
        newf = self.alignment_cost(newh, q2k)
        cost_evaluations = 1

        # backtrack while the Armijo criterion is not satisfied
        # or if newh goes outside positive orthant
        tst = newh<=0
        while (ls_backtrack and ((newf > (f0 + suff_decr*alpha*df0)) or (tst.sum>0))):
            # reduce the step size
            alpha *= contraction_factor

            # look closer down the line
            newh = self.exp(hid, d, alpha)
            newf = self.alignment_cost(newh, q2k)
            cost_evaluations += 1

            # make sure we don't run out of budget
            if cost_evaluations >= max_ls_steps:
                break
        
        # if we got here with obtaining a derease, reject the step
        if ls_force_decrease and newf > f0:
            alpha = 0
            newh = hid
            newf = f0
        
        # As seen outside this function, stepsize is the size of the vector we
        # retract to make the step from h to newh. Since the step is alpha*d:
        norm_d = self.norm(d)
        stepsize = alpha * norm_d

        # return some statistics
        lsstats = {"costevals":cost_evaluations,"stepsize":stepsize,"alpha":alpha}

        return stepsize, newh, lsstats

    def stoppingcriterion(self, options, info, last):
        stop = 0
        stats = info[last]
        if stats['gradnorm'] <= options["tolgradnorm"]:
            stop = 2
        
        if stats['time'] >= options["maxtime"]:
            stop = 3
        
        if stats['iter'] >= options["maxiter"]:
            stop = 4

        return stop

    def group_action_SRVF(self, q, h):
        p = q.shape[0]
        gamma = np.zeros(self.T)
        gamma[1:] = cumtrapz(h**2,self.t)
        h = np.sqrt(np.gradient(gamma,self.t))
        qnew = q
        if q.ndim > 1:
            for i in range(0,p):
                qnew[i,:] = np.interp(gamma,self.t,q[i,:])*h
        else:
            qnew = np.interp(gamma,self.t,q)*h

        return qnew

    def normL2(self, f):
        val = np.sqrt(self.innerProdL2(f,f))
        return val

    def innerProdL2(self,f1,f2):
        tmp = f1*f2
        val = trapz(tmp.sum(axis=0),self.t)
        return val

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


    

