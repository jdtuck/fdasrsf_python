"""
statistic calculation for SRVF (curves) open and closed using Karcher
Mean and Variance

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""
from numpy import zeros, sqrt, fabs, cos, sin, tile, vstack, empty, cov, inf, mean, arange, prod
from numpy import append, arccos, cumsum, sum
from numpy.linalg import svd
from numpy.random import randn
import fdasrsf.curve_functions as cf
import fdasrsf.utility_functions as uf
import fdasrsf.plot_style as plot
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
    :param q:           (n,T,K) matrix defining n dimensional srvf on T samples with K srvfs
    :param betan:       aligned curves
    :param qn:          aligned srvfs
    :param basis:       calculated basis
    :param beta_mean:   karcher mean curve
    :param q_mean:      karcher mean srvf
    :param gams:        warping functions
    :param v:           shooting vectors
    :param C:           karcher covariance
    :param s:           pca singular values
    :param U:           pca singular vectors
    :param coef:        pca coefficients
    :param pca          principal directions
    :param qun:         cost function
    :param samples:     random samples
    :param gamr:        random warping functions
    :param cent:        center
    :param scale:       scale
    :param len:         length of curve
    :param len_q:       length of srvf
    :param mean_scale   mean length
    :param mean_scale_q mean length srvf
    :param E:           energy

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  26-Aug-2020
    """

    def __init__(self, beta, mode='O', N=200, scale=False):
        """
        fdacurve Construct an instance of this class
        :param beta: (n,T,K) matrix defining n dimensional curve on T samples with K curves
        :param mode:  Open ('O') or closed curve ('C') (default 'O')
        :param N: resample curve to N points
        :param scale: include scale (true/false)
        """
        self.mode = mode
        self.scale = scale

        K = beta.shape[2]
        n = beta.shape[0]
        q = zeros((n,N,K))
        beta1 = zeros((n,N,K))
        cent1 = zeros((n,K))
        len1 = zeros(K)
        lenq1 = zeros(K)
        for ii in range(0,K):
            if (beta.shape[1] != N):
                beta1[:,:,ii] = cf.resamplecurve(beta[:,:,ii],N,mode=mode)
            else:
                beta1[:,:,ii] = beta[:,:,ii]
            a = -cf.calculatecentroid(beta1[:,:,ii])
            beta1[:,:,ii] += tile(a, (N,1)).T
            q[:,:,ii], len1[ii], lenq1[ii] = cf.curve_to_q(beta1[:,:,ii], mode)
            cent1[:,ii] = -a

        self.q = q
        self.beta = beta1
        self.cent = cent1
        self.len = len1
        self.len_q = lenq1


    def karcher_mean(self, rotation=True, parallel=False, cores=-1, method="DP"):
        """
        This calculates the mean of a set of curves
        :param rotation: compute optimal rotation (default = T)
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
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

        gamma = zeros((T,N))
        maxit = 20

        sumd = zeros(maxit+1)
        v = zeros((n, T, N))
        sumv = zeros((n,T))
        normvbar = zeros(maxit+1)

        delta = 0.5
        tolv = 1e-4
        told = 5*1e-3

        print("Computing Karcher Mean of %d curves in SRVF space.." % N)
        while itr < maxit:
            print("updating step: %d" % (itr+1))

            if iter == maxit:
                print("maximal number of iterations reached")

            mu = mu / sqrt(cf.innerprod_q2(mu, mu))
            if mode == 1:
                self.basis = cf.find_basis_normal(mu)
            else:
                self.basis = []

            sumv = zeros((n, T))
            sumd[0] = inf
            sumd[itr+1] = 0
            out = Parallel(n_jobs=cores)(delayed(karcher_calc)(mu, self.q[:, :, n], self.basis, 
                                                               mode, rotation, method) for n in range(N))
            v = zeros((n, T, N))
            gamma = zeros((T,N))
            for i in range(0, N):
                v[:, :, i] = out[i][0]
                gamma[:, i] = out[i][1]
                sumv += v[:, :, i]
                sumd[itr+1] = sumd[itr+1] + out[i][2]**2

            sumv = v.sum(axis=2)

            # Compute average direction of tangent vectors v_i
            vbar = sumv/float(N)

            normvbar[itr] = sqrt(cf.innerprod_q2(vbar, vbar))
            normv = normvbar[itr]

            if ((sumd[itr]-sumd[itr+1]) < 0):
                break
            elif (normv > tolv and fabs(sumd[itr+1]-sumd[itr]) > told):
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

        # compute average length
        if self.scale:
            self.mean_scale = (prod(self.len))**(1./self.len.shape[0])
            self.mean_scale_q = (prod(self.len_q))**(1./self.len.shape[0])
            betamean = self.mean_scale * betamean
        
        self.q_mean = mu
        self.beta_mean = betamean
        self.v = v
        self.qun = sumd[0:(itr+1)]
        self.E = normvbar[0:(itr+1)]

        return


    def srvf_align(self, rotation=True, parallel=False, cores=-1, method="DP"):
        """
        This aligns a set of curves to the mean and computes mean if not computed
        :param rotation: compute optimal rotation (default = T)
        :param parallel: run in parallel (default = F)
        :param cores: number of cores for parallel (default = -1 (all))
        :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
        """
        n, T, N = self.beta.shape

        modes = ['O', 'C']
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        # find mean
        if not hasattr(self, 'beta_mean'):
            self.karcher_mean()

        self.qn = zeros((n, T, N))
        self.betan = zeros((n, T, N))
        self.gams = zeros((T,N))
        centroid2 = cf.calculatecentroid(self.beta_mean)
        self.beta_mean = self.beta_mean - tile(centroid2, [T, 1]).T

        # align to mean
        out = Parallel(n_jobs=-1)(delayed(cf.find_rotation_and_seed_unique)(self.q_mean,
                                          self.q[:, :, n], mode, rotation, method) for n in range(N))
        for ii in range(0, N):
            self.gams[:,ii] = out[ii][2]
            self.qn[:, :, ii] = out[ii][0]
            btmp = out[ii][1].dot(self.beta[:,:,ii])
            self.betan[:, :, ii] = cf.group_action_by_gamma_coord(btmp,out[ii][2])

        return


    def karcher_cov(self):
        """
        This calculates the mean of a set of curves

        """
        if not hasattr(self, 'beta_mean'):
            self.karcher_mean()

        M,N,K = self.v.shape

        N1 = M*N
        if (self.scale):
            N1 += 1

        tmpv = zeros((N1,K))
        for i in range(0,K):
            tmp = self.v[:,:,i]
            tmpv1 = tmp.flatten()
            if (self.scale):
                tmpv1 = append(tmpv1, self.len[i])

            tmpv[:,i] = tmpv1

        if (self.scale):
            VM = mean(self.v, 2)
            VM = VM.flatten()
            VM = append(VM, self.mean_scale)
            tmpv = tmpv - tile(VM, [K, 1]).T
            self.C = cov(tmpv)
        else:
            self.C = cov(tmpv)

        return


    def shape_pca(self, no=10):
        """
        Computes principal direction of variation specified by no. N is
        Number of shapes away from mean. Creates 2*N+1 shape sequence

        :param no: number of direction (default 3)
        """
        if not hasattr(self, 'C'):
            self.karcher_cov()

        U1, s, V = svd(self.C)
        self.U = U1[:,0:no]
        self.s = s[0:no]

        # express shapes as coefficients
        K = self.beta.shape[2]
        VM = mean(self.v,2)
        VM = VM.flatten()
        if (self.scale):
            VM = append(VM, self.mean_scale)

        x = zeros((no,K))
        for ii in range(0,K):
            tmpv = self.v[:,:,ii]
            tmpv1 = tmpv.flatten()
            if (self.scale):
                tmpv1 = append(tmpv1, self.len[ii])
            Utmp = self.U.T
            x[:,ii] = Utmp.dot((tmpv1- VM))

        self.coef = x

        modes = ['O', 'C']
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        n1, T, N1 = self.beta.shape
        p = zeros((n1,T,no,10))
        for j in range(0,no):
            for i in range(1,11):
                tmp = VM + 0.5*(i-5)*sqrt(self.s[j])*self.U[:,j]
                m,n = self.q_mean.shape
                if (self.scale):
                    tmp_scale = tmp[-1]
                    tmp = tmp[0:-1]
                else:
                    tmp_scale = 1
                v1 = tmp.reshape(m,n)
                q2n = cf.elastic_shooting(self.q_mean, v1, mode)

                p[:,:,j,i-1] = cf.q_to_curve(q2n, tmp_scale)            

        self.pca = p

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
            ax.plot(self.beta[0,:,ii],self.beta[1,:,ii])
        plt.title('Curves')
        ax.set_aspect('equal')
        plt.axis('off')
        plt.gca().invert_yaxis()

        if hasattr(self,'gams'):
            M = self.gams.shape[0]
            fig, ax = plot.f_plot(arange(0, M) / float(M - 1), self.gams,
                                    title="Warping Functions")

        if hasattr(self,'beta_mean'):
            fig, ax = plt.subplots()
            ax.plot(self.beta_mean[0,:],self.beta_mean[1,:])
            plt.title('Karcher Mean')
            ax.set_aspect('equal')
            plt.axis('off')
            plt.gca().invert_yaxis()

        plt.show()


    def plot_pca(self):

        if not hasattr(self,'s'):
            raise NameError('Calculate PCA')

        fig, ax = plt.subplots()
        ax.plot(cumsum(self.s)/sum(self.s)*100)
        plt.title('Variability Explained')
        plt.xlabel('PC')

        # plot principal modes of variability
        VM = mean(self.v,2)
        VM = VM.flatten()
        if (self.scale):
            VM = append(VM, self.mean_scale)

        modes = ['O', 'C']
        mode = [i for i, x in enumerate(modes) if x == self.mode]
        if len(mode) == 0:
            mode = 0
        else:
            mode = mode[0]

        for j in range(0,4):
            fig, ax = plt.subplots()
            for i in range(1,11):
                tmp = VM + 0.5*(i-5)*sqrt(self.s[j])*self.U[:,j]
                m,n = self.q_mean.shape
                if (self.scale):
                    tmp_scale = tmp[-1]
                    tmp = tmp[0:-1]
                else:
                    tmp_scale = 1
                v1 = tmp.reshape(m,n)
                q2n = cf.elastic_shooting(self.q_mean,v1,mode)

                p = cf.q_to_curve(q2n, tmp_scale)

                mv = 0.2
                if (self.scale):
                    mv *= self.mean_scale
                
                if i == 5:
                    ax.plot(mv*i + p[0,:], p[1,:], 'k', linewidth=2)
                else:
                    ax.plot(mv*i + p[0,:], p[1,:], linewidth=2)

            ax.set_aspect('equal')
            plt.axis('off')
            plt.title('PD %d' % (j+1))

        plt.show()


def karcher_calc(mu, q, basis, closed, rotation, method):
    # Compute shooting vector from mu to q_i
    qn_t, R, gamI = cf.find_rotation_and_seed_unique(mu, q, closed, rotation, method)
    qn_t = qn_t / sqrt(cf.innerprod_q2(qn_t,qn_t))

    q1dotq2 = cf.innerprod_q2(mu,qn_t)

    if (q1dotq2 > 1):
        q1dotq2 = 1
    
    d = arccos(q1dotq2)

    u = qn_t - q1dotq2*mu
    normu = sqrt(cf.innerprod_q2(u,u))
    if (normu>1e-4):
        w = u*arccos(q1dotq2)/normu
    else:
        w = zeros(qn_t.shape)

    # Project to tangent space of manifold to obtain v_i
    if closed == 0:
        v = w
    else:
        v = cf.project_tangent(w, q, basis)

    return(v, gamI, d)
