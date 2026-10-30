"""
Utility functions for SRSF Manipulations

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import time
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from numpy.linalg import norm


class rlbfgs:
    r"""
    This class provides alignment methods for functional data using the SRVF
    framework using the Riemannian limited memory BFGS solver, following
    Huang, Gallivan, Srivastava and Absil, "Riemannian optimization for
    registration of curves in elastic shape analysis", for the warping alone
    (no rotation and no shift).

    The iterate is :math:`l=\sqrt{\dot{\gamma}}` on the unit sphere of
    :math:`L^2([0,1],R)`, :math:`\gamma` = cumtrapz(:math:`l^2`), and the cost
    is :math:`||q_1 - q_2(\gamma) l||^2` plus the penalty plus the paper's
    barrier :math:`B(l) = \int (l^2 + 1/l^2) \sqrt{1 + l^4}` (their eq. (15)),
    whose weight omega starts at 1/100 and is cut by 0.8 each iteration.
    Steps are taken with the exponential map, and the stored BFGS pairs are
    moved to the new tangent space by parallel translation (their Lemma 1).
    q2 is represented by a natural cubic spline, so the cost is twice
    differentiable, and the gradient differentiates the discretised cost
    exactly.

    The inputs q1 and q2 are the square root velocity functions of curves in
    R^n to be aligned. Here, q2 will be aligned to q1.

    Usage:  obj = rlbfgs(q1,q2,t)

    :param q1: (M,N): matrix defining srvf of dimension M of N samples
    :param q2: (M,N): matrix defining srvf of dimension M of N samples
    :param t: time vector of length N
    :param q2Opt: optimally aligned srvf
    :param gammaOpt: optimal warping function
    :param cost: final cost
    :param info: list of dictionaries consisting of info about the iterations

    Wen Huang, Kyle A. Gallivan, Anuj Srivastava, Pierre-Antoine Absil.
    "Riemannian Optimization for Elastic Shape Analysis", Short version,
    The 21st International Symposium on Mathematical Theory of Networks and
    Systems (MTNS 2014).

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

        self.t = t
        self.T = t.shape[0]
        if q1.ndim > 1:
            self.q1 = q1
            self.q2 = q2
            self.lamScale = 1.0
        else:
            # unit L2 norm on t rather than unit Euclidean norm, so the cost,
            # its gradient and so tolgradnorm do not scale with the grid size
            self.q1 = q1 / self.normL2(q1)
            self.q2 = q2 / self.normL2(q2)
            # the data term used to be that of the unit Euclidean norm SRVFs;
            # lamScale keeps lam weighing the penalty against it as before
            self.lamScale = 1 / (
                self.normL2(q1 / norm(q1)) * self.normL2(q2 / norm(q2))
            )

        # q2 is composed with gamma, so it is represented by an interpolatory
        # cubic spline, which makes the cost twice differentiable as the
        # convergence theory of the quasi-Newton method needs
        self.q2spline = CubicSpline(t, self.q2, axis=-1, bc_type="natural")
        self.w = self.trapz_weights(t)

    def solve(self, maxiter=30, verb=0, lam=0, penalty="roughness", gam0=None):
        """
        Run solver

        :param maxiter: maximum number of interations
        :param verb: integer used to tune the amount of output
        :param lam: amount of penalty
        :param penalty: penalty, "roughness", "l2gam", "l2psi", "geodesic"
        :param gam0: warping function sampled on t to start from (default =
                     None, the identity)

        After solving, ``stopReason`` holds why the solver stopped (1 no
        acceptable step, 2 gradient norm, 3 iterations, 4 relative change of
        the cost), ``iterations`` the iterations run and ``gradNorm`` the
        final gradient norm.
        """
        if penalty not in ("roughness", "l2gam", "l2psi", "geodesic"):
            raise Exception("penalty not implemented")
        lam = lam * self.lamScale

        # a safeguard; the paper's own test is on the change of the cost
        tolgradnorm = 1e-10
        # number of previous iterations the program remembers
        memory = 30
        ls_max_steps = 25
        # the paper stops once the relative change of the cost has stayed
        # below 1e-3 for more than 5 iterations, after at least 10
        miniter = 10
        tolrelcost = 1e-3
        relcostcount = 5
        omega = 1.0 / 100
        omegarate = 0.8

        timetic = time.time()
        l = np.ones(self.T)
        if gam0 is not None:
            gam0 = np.asarray(gam0, dtype=float)
            if gam0.shape[0] != self.T:
                raise ValueError("gam0 must have as many elements as t")
            l = self.srvf_of_warp(gam0)
        l = l / self.norm(l)

        sHistory = []
        yHistory = []
        rhoHistory = []

        # Total number of BFGS iterations
        k = 0
        # scaling of initial matrix, Barzilai-Borwein
        scaleFactor = 1
        stop = 0
        smallchange = 0
        failedsteps = 0

        cur = self.costgrad(l, lam, penalty)
        f = cur[0] + omega * cur[1]
        g = self.riemannian_grad(l, cur, omega)
        gnorm = self.norm(g)

        if verb >= 2:
            print(" iter                   cost val            grad. norm\n")

        info = [{"iter": k, "cost": cur[0], "gradnorm": gnorm,
                 "time": time.time() - timetic}]

        while True:
            if verb >= 2:
                print("%5d    %+.16e        %.8e\n" % (k, f, gnorm))

            if gnorm <= tolgradnorm:
                stop = 2
                break
            if k >= maxiter:
                stop = 3
                break

            # compute BFGS direction; fall back to steepest descent if the
            # quasi-Newton direction is not a descent direction
            p = self.getDirection(g, sHistory, yHistory, rhoHistory, scaleFactor)
            df0 = self.inner(g, p)
            if df0 >= 0:
                sHistory, yHistory, rhoHistory = [], [], []
                scaleFactor = 1
                p = -g
                df0 = -gnorm * gnorm

            # Armijo backtracking along the exponential map
            alpha = 1
            lnew = self.exp(l, p, alpha)
            nxt = self.costgrad(lnew, lam, penalty)
            fnew = nxt[0] + omega * nxt[1]
            evals = 1
            while (
                np.any(lnew <= 0)
                or not np.isfinite(fnew)
                or fnew > f + 1e-4 * alpha * df0
            ):
                alpha *= 0.5
                lnew = self.exp(l, p, alpha)
                nxt = self.costgrad(lnew, lam, penalty)
                fnew = nxt[0] + omega * nxt[1]
                evals += 1
                if evals >= ls_max_steps:
                    break
            if np.any(lnew <= 0) or not np.isfinite(fnew) or fnew > f:
                # no acceptable step: restart the memory once, then stop
                failedsteps += 1
                if failedsteps >= 2 or len(sHistory) == 0:
                    stop = 1
                    break
                sHistory, yHistory, rhoHistory = [], [], []
                scaleFactor = 1
                continue
            failedsteps = 0

            # the gradient at the new point with this iteration's omega, so
            # that the BFGS pair is that of one cost function
            gnew = self.riemannian_grad(lnew, nxt, omega)

            # parallel translation of tangent vectors from l to lnew
            sk = self.transp(l, lnew, alpha * p)
            yk = gnew - self.transp(l, lnew, g)
            sHistory = [self.transp(l, lnew, s) for s in sHistory]
            yHistory = [self.transp(l, lnew, y) for y in yHistory]

            # cautious update
            inner_sk_yk = self.inner(sk, yk)
            inner_sk_sk = self.inner(sk, sk)
            if inner_sk_sk > 0 and inner_sk_yk / inner_sk_sk >= 1e-4 * gnorm:
                if len(sHistory) >= memory:
                    sHistory.pop(0)
                    yHistory.pop(0)
                    rhoHistory.pop(0)
                sHistory.append(sk)
                yHistory.append(yk)
                rhoHistory.append(1 / inner_sk_yk)
                scaleFactor = inner_sk_yk / self.inner(yk, yk)

            # relative change of the cost, both values at this omega
            relchange = abs(f - fnew) / max(abs(f), 1e-300)
            smallchange = smallchange + 1 if relchange < tolrelcost else 0

            l = lnew
            cur = nxt
            k += 1

            # the barrier fades as the iterations proceed
            omega *= omegarate
            f = cur[0] + omega * cur[1]
            g = self.riemannian_grad(l, cur, omega)
            gnorm = self.norm(g)

            info.append({"iter": k, "cost": cur[0], "gradnorm": gnorm,
                         "time": time.time() - timetic})

            if k >= miniter and smallchange > relcostcount:
                stop = 4
                break

        gam = np.zeros(self.T)
        gam[1:] = cumulative_trapezoid(l**2, self.t)
        self.gammaOpt = (gam - gam.min()) / (gam.max() - gam.min())
        self.q2Opt = self.q2spline(gam) * l
        self.info = info
        # the cost of the warping returned, without the barrier
        self.cost = cur[0]
        self.stopReason = stop
        self.iterations = k
        self.gradNorm = gnorm

        if verb >= 1:
            print("Total time is %f [s]\n" % (time.time() - timetic))

        return

    def penalty_cost(self, h, penalty):
        """
        Value of the penalty term at the warping h = sqrt(gamma'), the total
        warping from the identity
        """
        time1 = np.linspace(0, 1, h.shape[0])
        if penalty == "roughness":
            binsize = np.mean(np.diff(time1))
            pen = trapezoid(np.gradient(h**2, binsize) ** 2, time1)
        elif penalty == "l2gam":
            pen = self.normL2(h**2 - 1) ** 2
        elif penalty == "l2psi":
            pen = self.normL2(h - 1) ** 2
        elif penalty == "geodesic":
            q1dotq2 = np.clip(trapezoid(h, time1), -1, 1)
            pen = np.arccos(q1dotq2) ** 2
        else:
            raise Exception("penalty not implemented")

        return pen

    def penalty_grad(self, h, penalty):
        """
        Partial derivatives of penalty_cost with respect to each sample of h.
        These differentiate the discretised penalties exactly, so they agree
        with penalty_cost on the grid and not merely in the continuum limit
        """
        time1 = np.linspace(0, 1, h.shape[0])
        w = self.trapz_weights(time1)
        if penalty == "roughness":
            # pen = trapz(d(h^2)/dt ^ 2), through gradient() and the square
            binsize = np.mean(np.diff(time1))
            hdot = np.gradient(h**2, binsize)
            dP = 2 * h * self.gradient_adjoint(2 * w * hdot, binsize)
        elif penalty == "l2gam":
            dP = 4 * h * (h**2 - 1) * w
        elif penalty == "l2psi":
            dP = 2 * (h - 1) * w
        elif penalty == "geodesic":
            # acos(c)/sqrt(1-c^2) tends to 1 as c tends to 1, where both
            # factors vanish
            c = np.clip(trapezoid(h, time1), -1, 1)
            scale = 1.0
            if 1 - c**2 > 1e-12:
                scale = np.arccos(c) / np.sqrt(1 - c**2)
            dP = -2 * scale * w
        else:
            raise Exception("penalty not implemented")

        return dP

    def costgrad(self, l, lam, penalty):
        """
        Cost parts (data and penalty, barrier) and their partial derivatives
        with respect to each sample of l, differentiating the discretised cost
        exactly: gamma is cumtrapz(l^2), q2 is evaluated on its cubic spline
        and the integrals are trapezoidal sums
        """
        w = self.w
        gam = np.zeros(self.T)
        gam[1:] = cumulative_trapezoid(l**2, self.t)
        gam = np.clip(gam, self.t[0], self.t[-1])
        Q = self.q2spline(gam)
        Qdot = self.q2spline(gam, 1)
        r = self.q1 - Q * l
        if r.ndim > 1:
            fixedcost = np.sum(w * r * r)
            # r depends on l directly and through gamma
            a = -2 * w * np.sum(r * Qdot, axis=0) * l
            fixedgrad = -2 * w * np.sum(r * Q, axis=0)
        else:
            fixedcost = np.sum(w * r * r)
            a = -2 * w * r * Qdot * l
            fixedgrad = -2 * w * r * Q
        fixedgrad = fixedgrad + 2 * l * self.cumtrapz_adjoint(self.t, a)

        if lam != 0:
            fixedcost += lam * self.penalty_cost(l, penalty)
            fixedgrad = fixedgrad + lam * self.penalty_grad(l, penalty)

        # B = trapz((l^2 + 1/l^2) sqrt(1 + l^4)), eq. (15) of the paper; it
        # grows where gamma is nearly flat or vertical and is symmetric under
        # inverting gamma
        l2 = l**2
        rt = np.sqrt(1 + l2**2)
        u = l2 + 1 / l2
        barrier = np.sum(w * u * rt)
        du = 2 * l - 2 / l**3
        drt = 2 * l**3 / rt
        barriergrad = w * (du * rt + u * drt)

        return fixedcost, barrier, fixedgrad, barriergrad

    def riemannian_grad(self, l, c, omega):
        """
        Riemannian gradient under the L2 metric inner() uses: the partial
        derivatives divided by the quadrature weights, projected onto the
        tangent space of the sphere at l
        """
        v = (c[2] + omega * c[3]) / self.w
        return v - self.inner(v, l) * l

    def srvf_of_warp(self, gam):
        """
        l = sqrt(gamma') of the warping gam, sampled on t, from central
        differences; a small floor keeps l inside the barrier
        """
        gam = (gam - gam[0]) / (gam[-1] - gam[0])
        return np.sqrt(np.clip(np.gradient(gam, 1.0 / (self.T - 1)), 1e-6, None))

    def getDirection(self, g, sHistory, yHistory, rhoHistory, scaleFactor):
        """
        BFGS step, see Wen's paper for details. This function takes in a
        tangent vector g, and applies an approximate inverse Hessian P to
        it to get Pg. Then, -Pg is returned.  The stored pairs have been
        parallel translated to the tangent space of the current iterate.
        """
        q = g
        j = len(sHistory)
        inner_s_q = np.zeros(j)

        for i in range(j, 0, -1):
            inner_s_q[i - 1] = rhoHistory[i - 1] * self.inner(sHistory[i - 1], q)
            q = q - inner_s_q[i - 1] * yHistory[i - 1]

        r = scaleFactor * q

        for i in range(0, j):
            om = rhoHistory[i] * self.inner(yHistory[i], r)
            r = r + (inner_s_q[i] - om) * sHistory[i]

        return -r

    def transp(self, l, lnew, v):
        """
        Parallel translation of v from the tangent space at l to that at
        lnew = exp(l, u), Lemma 1 of the paper
        """
        s = l + lnew
        n2 = self.inner(s, s)
        if n2 <= 0:
            return v
        return v - (2 * self.inner(v, lnew) / n2) * s

    def trapz_weights(self, x):
        """
        Trapezoidal quadrature weights for the grid x, so that
        trapezoid(y, x) equals sum(w * y)
        """
        w = np.zeros(x.shape[0])
        dx = np.diff(x) / 2.0
        w[:-1] += dx
        w[1:] += dx
        return w

    def gradient_adjoint(self, y, binsize):
        """
        Adjoint of np.gradient (edge_order=1) on a uniform grid
        """
        out = np.zeros(y.shape[0])
        out[1] += y[0] / binsize
        out[0] -= y[0] / binsize
        out[-1] += y[-1] / binsize
        out[-2] -= y[-1] / binsize
        out[2:] += y[1:-1] / (2 * binsize)
        out[:-2] -= y[1:-1] / (2 * binsize)
        return out

    def cumtrapz_adjoint(self, x, b):
        """
        Adjoint of the running trapezoidal sum, built from the suffix sums of b
        """
        acc = np.cumsum(b[::-1])[::-1][1:]
        c = np.diff(x) / 2.0 * acc
        out = np.zeros(x.shape[0])
        out[1:] += c
        out[:-1] += c
        return out

    def normL2(self, f):
        val = np.sqrt(self.innerProdL2(f, f))
        return val

    def innerProdL2(self, f1, f2):
        tmp = f1 * f2
        if tmp.ndim > 1:
            val = trapezoid(tmp.sum(axis=0), self.t)
        else:
            val = trapezoid(tmp, self.t)
        return val

    def dist(self, f1, f2):
        d = np.real(np.arccos(np.clip(self.inner(f1, f2), -1, 1)))
        return d

    def typicaldist(self):
        return np.pi / 2

    def proj(self, f, v):
        out = v - f * trapezoid(f * v, self.t)
        return out

    def log(self, f1, f2):
        v = self.proj(f1, f2 - f1)
        di = self.dist(f1, f2)
        if di > 1e-6:
            nv = self.norm(v)
            v = v * (di / nv)

        return v

    def exp(self, f1, v, delta=1):
        vd = delta * v
        nrm_vd = self.norm(vd)

        if nrm_vd > 0:
            f2 = f1 * np.cos(nrm_vd) + vd * (np.sin(nrm_vd) / nrm_vd)
        else:
            f2 = f1

        return f2

    def inner(self, v1, v2):
        return trapezoid(v1 * v2, self.t)

    def norm(self, v):
        return np.sqrt(trapezoid(v**2, self.t))

    def zerovec(self):
        return np.zeros(self.T)
