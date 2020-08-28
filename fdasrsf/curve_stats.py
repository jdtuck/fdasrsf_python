"""
statistic calculation for SRVF (curves) open and closed using Karcher
Mean and Variance

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty, cov, inf, mean
from numpy.linalg import svd
from numpy.random import randn
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import collections

class fdacurve:
    """
    This class provides alignment methods for open and closed curves using the SRVF framework

    Usage: obj = fdacurve(beta, mode, N, scale)
    :param beta: numpy ndarray of shape (n, M, N) describing N curves
    in R^M
    :param mode: Open ('O') or closed curve ('C') (default 'O')
    :param N: resample curve to N points
    :param scale: scale curve to length 1 (true/false)
    :param q:        (n,T,K) matrix defining n dimensional srvf on T samples with K srvfs
    :param betan:     aligned curves
    :param qn:        aligned srvfs
    :param basis:     calculated basis
    :param beta_mean: karcher mean curve
    :param q_mean:    karcher mean srvf
    :param gams:      warping functions
    :param v:         shooting vectors
    :param C:         karcher covariance
    :param s:         pca singular values
    :param U:         pca singular vectors
    :param coef:      pca coefficients
    :param qun:       cost function
    :param samples:   random samples
    :param gamr:      random warping functions
    :param cent:      center
    :param scale:     scale
    :param E:         energy

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  26-Aug-2020
    """

    def __init__(self, beta, mode='O', N=200, scale=True):
        """
        fdacurve Construct an instance of this class
        :param beta: (n,T,K) matrix defining n dimensional curve on T samples with K curves
        :param mode:  Open ('O') or closed curve ('C') (default 'O')
        :param N: resample curve to N points
        :param scale: scale curve to length 1 (true/false)
        """
        self.mode = mode
        self.scale = scale

        K = beta.shape[2]
        n = beta.shape[0]
        q = zeros((n,N,K))
        beta1 = zeros((n,N,K))
        cent1 = zeros((n,K))
        for ii in range(0,K):
            beta1[:,:,ii] = cf.resamplecurve(beta[:,:,ii],N,mode)
            a = -cf.calculatecentroid(beta1[:,:,ii])
            beta1[:,:,ii] += tile(a, (N,1)).T
            q[:,:,ii] = cf.curve_to_q(beta1[:,:,ii], self.scale, self.mode)
            cent1[:,ii] = -a
        
        self.q = q
        self.beta = beta
        self.cent = cent1


    def karcher_mean(self, parallel=False, cores=-1):
        """
        This calculates the mean of a set of curves
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        """
        n, T, N = self.beta.shape

        modes = ['O', 'C']
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # Initialize mu as one of the shapes
        mu = self.q[:, :, 0]
        betamean = self.beta[:,:,0]
        itr = 0
        T = mu.shape[1]
        N = mu.shape[0]
        K = self.q.shape[2]
        gamma = zeros((T,N))
        maxit = 20

        sumd = zeros(maxit+1)
        v = zeros((n, T, N))
        normvbar = zeros(maxit+1)

        delta = 0.5
        tolv = 1e-4
        told = 5*1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr < maxit:
            print("updating step: %d" % itr)

            if iter == maxit:
                print("maximal number of iterations reached")
            
            mu = mu / sqrt(cf.innerprod_q2(mu, mu))
            if mode == 1:
                self.basis = cf.find_basis_normal(mu)

            sumv = zeros((n, T))
            sumd[0] = inf
            sumd[itr+1] = 0
            out = Parallel(n_jobs=-1)(delayed(karcher_calc)(self.beta[:, :, n],
                                    self.q[:, :, n], betamean, mu, self.basis, mode) for n in range(N))
            v = zeros((n, T, N))
            for i in range(0, N):
                v[:, :, i] = out[i][0]
                sumd[itr+1] = sumd[itr+1] + out[i][1]**2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv/float(N)

            normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
            normv = normvbar[itr]

            if normv > tolv and fabs(sumd[itr+1]-sumd[itr]) > told:
                # Update mu in direction of vbar
                mu = cos(delta*normvbar[itr])*mu + sin(delta*normvbar[itr]) * vbar/normvbar[itr]

                if mode == 1:
                    mu = cf.project_curve(mu)

                x = cf.q_to_curve(mu)
                a = -1*cf.calculatecentroid(x)
                betamean = x + tile(a, [T, 1]).T
            else:
                break

            itr += 1
        
        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0:itr]
        self.E = normvbar[0:(itr-1)]

        return


    def srvf_align(self):
        """
        This calculates the mean of a set of curves and aligns them
        """
        n, T, N = self.beta.shape
        # find mean
        if not hasattr(self, 'beta_mean'):
            self.karcher_mean()

        self.qn = zeros((n, T, N))
        self.betan = zeros((n, T, N))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - tile(centroid2, [T, 1]).T
        q_mu = cf.curve_to_q(self.beta_mean)
        # align to mean
        for ii in range(0, N):
            beta1 = self.beta[:, :, ii]

            # Iteratively optimize over SO(n) x Gamma
            for i in range(0, 1):
                # Optimize over SO(n)
                beta1, O_hat, tau = cf.find_rotation_and_seed_coord(self.beta_mean,
                                                                    beta1)
                q1 = cf.curve_to_q(beta1)

                # Optimize over Gamma
                gam = cf.optimum_reparam_curve(q1, q_mu, 0.0)
                gamI = uf.invertGamma(gam)
                # Applying optimal re-parameterization to the second curve
                beta1 = cf.group_action_by_gamma_coord(beta1, gamI)

            # Optimize over SO(n)
            beta1, O_hat, tau = cf.find_rotation_and_seed_coord(self.beta_mean, beta1)
            self.qn[:, :, ii] = cf.curve_to_q(beta1)
            self.betan[:, :, ii] = beta1

        return


    def karcher_cov(self):
        """
        This calculates the mean of a set of curves

        """
        if not hasattr(self, 'beta_mean'):
            self.karcher_mean()
        M,N,K = self.v.shape
        tmpv = zeros((M*N,K))
        for i in range(0,K):
            tmp = self.v[:,:,i]
            tmpv[:,i] = tmp.flatten()

        self.C = cov(tmpv.T)

        return


    def shape_pca(self, no=3, N=5):
        """
        Computes principal direction of variation specified by no. N is
        Number of shapes away from mean. Creates 2*N+1 shape sequence

        :param no: number of direction (default 3)
        :param N: number of shapes (2*N+1) (default 5)
        """
        if not hasattr(self, 'C'):
            self.karcher_cov()

        U1, s, V = svd(self.C)
        self.U = U1[:,0:no]
        self.s = s[0:no]

        # express shapes as coefficients
        K = self.beta.shape[2]
        VM = mean(self.v,2)
        x = zeros((no,K))
        for ii in range(0,K):
            tmpv = self.v[:,:,ii]
            x[:,ii] = self.U.dot((tmpv.flatten()- VM.flatten()))
        
        self.coef = x

        return


    def sample_shapes(self, no=3, numSamp=10):
        """
        Computes sample shapes from mean and covariance

        :param no: number of direction (default 3)
        :param numSamp: number of samples (default 10)
        """
        n, T = self.q_mean.shape
        modes = ['O', 'C']
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        U, s, V = svd(self.C)

        if mode == 0:
            N = 2
        else:
            N = 10

        epsilon = 1./(N-1)

        samples = empty(numSamp, dtype=object)
        for i in range(0, numSamp):
            v = zeros((2, T))
            for m in range(0, no):
                v = v + randn()*sqrt(s[m])*vstack((U[0:T, m], U[T:2*T, m]))

            q1 = self.q_mean
            for j in range(0, N-1):
                normv = sqrt(cf.innerprod_q2(v, v))

                if normv < 1e-4:
                    q2 = self.q_mean
                else:
                    q2 = cos(epsilon*normv)*q1+sin(epsilon*normv)*v/normv
                    if mode == 1:
                        q2 = cf.project_curve(q2)

                # Parallel translate tangent vector
                basis2 = cf.find_basis_normal(q2)
                v = cf.parallel_translate(v, q1, q2, basis2, mode)

                q1 = q2

            samples[i] = cf.q_to_curve(q2)

        self.samples = samples
        return
    

    def plot(self):
        """
        plot curve mean results
        """
        fig, ax = plt.subplots()
        n,T,K = self.beta.shape
        for ii in range(0,K):
            ax.plot(self.beta[1,:,ii],self.beta[2,:,ii])
        plt.title('Curves')
        ax.set_aspect('equal')
        plt.axis('off')
        plt.gca().invert_yaxis()

        if hasattr(self,'beta_mean'):
            fig, ax = plt.subplots()
            ax.plot(self.beta_mean[1,:],self.beta_mean[2,:])
            plt.title('Karcher Mean')
            ax.set_aspect('equal')
            plt.axis('off')
            plt.gca().invert_yaxis()


def karcher_calc(beta, q, betamean, mu, basis, mode):
    # Compute shooting vector from mu to q_i
    w, d = cf.inverse_exp_coord(betamean, beta)

    # Project to tangent space of manifold to obtain v_i
    if mode == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return(v, d)
