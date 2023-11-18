"""
Elastic functional change point detection

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import matplotlib.pyplot as plt
import fdasrsf as fs
import fdasrsf.fPCA as fpca
import fdasrsf.utility_functions as uf
from scipy.linalg import norm, svd
from fdasrsf.geometry import L2norm

# Function Definitions
def BBridge(x = 0, y = 0, t0 = 0, T = 1, N = 100):
    if (T <= t0):
        raise Exception('Endpoint is earlier than beginning')
    
    dt = (T-t0)/N
    t = np.linspace(t0, T, N+1)
    rng = np.random.default_rng()
    samp = rng.standard_normal(N)
    X = np.insert(np.cumsum(samp)*np.sqrt(dt),0,0)
    BB = x + X - (t-t0)/(T-t0) * (X[N] - y + x)
    return BB


def LongRunCovMatrixPrecentered(mdobj, h=0, kern_type = "bartlett"): 
    D = mdobj.shape[0]
    N = mdobj.shape[1]
    
    def Kernel(i,h):
        x = i/h
        if kern_type == "flat":
            return 1
        if kern_type == "simple":
            return 0
        if kern_type == "bartlett":
            return 1-x
        if kern_type == "flat_top":
            if x < 0.1:
                return 1
            else:
                if (x >= 0.1 & x < 1.1):
                    return 1.1-x
                else:
                    return 0
        if kern_type == "parzen":
            if x < 0.5:
                return 1 - 6 * pow(x,2) + 6 * pow(abs(x),3)
            else:
                return 2 * pow((1 - abs(x)),3)
    D_mat = np.zeros((D, D))
    cdata = mdobj
    
    for k in range(0, D):
        for r in range(k, D):
            s = cdata[k,:] @ cdata[r,:]
            if h > 0:
                for i in range(h):
                    a = cdata[k,range(N-i)] @ cdata[r,range(i, N)]
                    a = a + cdata[r,range(N-i)] @ cdata[k,range(i, N)]
                    s = s + Kernel(i+1, h) * a
            D_mat[k,r] = s 
            D_mat[r,k] = D_mat[k,r]
    return D_mat/N


class elastic_change:
    """"
    This class provides elastic changepoint using elastic fpca

    Usage:  obj = elastic_change(f,time)
    
    :param f: (M,N) % matrix defining N functions of M samples
    :param time: time vector of length M
    :param BBridges: precomputed Brownian Bridges (default: None)
    :param use_BBridges: use precomputed Brownian Bridges (default: False)
    :param warp_data: aligned data (default: None)
    :param Sn: test statistic values
    :param Tn: max of test statistic 
    :param p: p-value
    :param k_star: change point
    :param values: values of computed Brownian Bridges
    :param dat_a: data before changepoint
    :param dat_b: data after changepoint
    :param warp_a: warping functions before changepoint
    :param warp_b: warping functions after changepoint
    :param mean_a: mean function before changepoint
    :param mean_b: mean function after changepoint
    :param warp_mean_a: mean warping function before changepoint
    :param warp_mean_b: mean warping function after changepoin

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  27-Apr-2022

    """

    def __init__(self, f, time, BBridges=None, use_BBridges=False, smooth_data=False, warp_data=None, use_warp_data = False, parallel= False, sparam=25):
        """
        Construct an instance of the elastic_change class
        :param f: (M,N) % matrix defining N functions of M samples
        :param time: time vector of length M
        :param BBridges: precomputed Brownian Bridges (default: None)
        :param use_BBridges: use precomputed Brownian Bridges (default: False)
        :param smooth_data: smooth function data (default: False)
        :param warp_data: precomputed aligned data (default: None)
        :param use_warp_data: use precomputed warping data (default: False)
        :param parallel: run computation in parallel (default: True)
        :param sparam: number of smoothing runs of box filter (default: 25)
        """
        self.f = f
        self.time = time
        self.BBridges = BBridges
        self.use_BBridges = use_BBridges
        
        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
            
        if use_warp_data: 
            self.warp_data = warp_data# Align Data
        else:
            self.warp_data = fs.fdawarp(self.f,self.time)
            self.warp_data.srsf_align(parallel=parallel)
    


    def compute(self, pca_method="vert", pc=0.95, d=1000, compute_epidemic=False, n_pcs=5, preset_pcs = False):
        """
        Compute elastic change detection
        :param pca_method: string specifying pca method (options = "combined",
                        "vert", or "horiz", default = "combined")
        :param pc: percentage of cumulative variance to use (default: 0.95)
        :param compute_epidemic: compute epidemic changepoint model (default: False)
        :param n_pcs: scalar specify number of principal components (default: 5)
        :param preset_pcs: use all PCs (default: TrFalseue)
        """
        N1 = self.f.shape[1]

        # Calculate PCA
        if pca_method=='combined':
            self.pca = fpca.fdajpca(self.warp_data)
        elif pca_method=='vert':
            self.pca = fpca.fdavpca(self.warp_data)
        elif pca_method=='horiz':
            self.pca = fpca.fdahpca(self.warp_data)
        else:
            raise Exception('Invalid fPCA Method')
        self.pca.calc_fpca(N1)
        cumm_coef = np.cumsum(self.pca.latent) / sum(self.pca.latent)
        if preset_pcs == True:
            self.no = n_pcs
        else:
            self.no = np.argwhere(cumm_coef>=pc)[0][0]

        lam = 1/self.pca.latent[0:self.no]
        Sigma = np.diag(lam)
        eta = self.pca.coef[:, 0:self.no]
        eta_bar = eta.sum(axis=0)/N1

        # Compute Test Statistic
        self.Sn = np.zeros(N1)
        for i in range(1,N1):
            tmp_eta = eta[0:i,:]
            tmp = tmp_eta.sum(axis=0)-tmp_eta.shape[0]*eta_bar
            
            self.Sn[i] = 1/(N1) * tmp.T@Sigma@tmp
        
        self.Tn = self.Sn.mean()
        self.k_star = self.Sn.argmax()

        # compute distribution
        if self.use_BBridges == False:
            values = np.zeros(d)
            for i in range(d):
                B_tmp = np.zeros((self.no,N1))
                for j in range(self.no):
                    B_tmp[j,:] = BBridge(N=N1-1)**2
                values[i] = B_tmp.sum(axis=0).mean()
        elif self.no > self.BBridges.shape[0]:
            values = np.zeros(d)
            for i in range(d):
                B_tmp = np.zeros((self.no,N1))
                for j in range(self.no):
                    B_tmp[j,:] = BBridge(N=N1-1)**2
                values[i] = B_tmp.sum(axis=0).mean()
        else:
            values = (self.BBridges[0:self.no, :, 0:d]**2).sum(axis=0).mean(axis =0)
            
            
    
        z = self.Tn <= values
        self.p = z[z==True].shape[0]/z.shape[0]
        
        self.sim_values = values

        self.dat_a = self.f[:,0:self.k_star]
        self.dat_b = self.f[:,self.k_star:]
        self.warp_a = self.warp_data.gam[:,0:self.k_star]
        self.warp_b = self.warp_data.gam[:,self.k_star:]
        self.mean_a = self.warp_data.fn[:,0:self.k_star].mean(axis=1)
        self.mean_b = self.warp_data.fn[:,self.k_star:].mean(axis=1)  
        if self.warp_a.size == 0:
            self.warp_mean_a = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_a)
            self.warp_mean_a = gam_mu
            
        if self.warp_b.size == 0:
            self.warp_mean_b = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_b)
            self.warp_mean_b = gam_mu
        
        self.delta = self.mean_b - self.mean_a
        if self.warp_mean_a.size > 0 and self.warp_mean_b.size > 0:
            self.delta_warp = self.warp_mean_b - self.warp_mean_a
        else:
            self.delta_warp = self.time
            
        if compute_epidemic: 
            self.compute_epidemic = True
            snk_minus_vals = np.zeros((N1, N1, self.no))
            snk_minus_summary = np.zeros((N1, N1))
            for i in range(N1):
                for j in range(N1):
                    if (i > j):
                        continue
                    tmp_eta = eta[0:i,:]
                    tmp = tmp_eta.sum(axis=0)-tmp_eta.shape[0]*eta_bar
                    tmp_eta_j = eta[0:j,:]
                    tmp_j = tmp_eta_j.sum(axis=0)-tmp_eta.shape[0]*eta_bar
                    snk_minus_vals[i,j,:] = tmp - tmp_j
                    snk_minus_summary[i,j] = sum(pow(snk_minus_vals[i,j], 2) * lam)
            self.tau1 = np.argmax(snk_minus_summary.max(axis = 1))
            self.tau2 = np.argmax(snk_minus_summary[self.tau1,:])
            self.Sn_epidemic = snk_minus_summary
            self.Tn_epidemic = 1/(pow(N1, 3)) * snk_minus_summary.sum()
             # compute distribution
                
                
            if self.use_BBridges == False:
                values = np.zeros(d)
                for i in range(d):
                    B_tmp = np.zeros((self.no,N1, N1))
                    for j in range(self.no):
                        BBridge_val = BBridge(N=N1-1)
                        for i1 in range(N1):
                            for j1 in range(N1):
                                if (i1 > j1):
                                    continue
                                B_tmp[j,i1, j1] = (BBridge_val[i1] - BBridge_val[j1])**2
                    values[i] = B_tmp.mean(axis = (1,2)).sum()
            elif self.no > self.BBridges.shape[0]:
                values = np.zeros(d)
                for i in range(d):
                    B_tmp = np.zeros((self.no,N1, N1))
                    for j in range(self.no):
                        BBridge_val = BBridge(N=N1-1)
                        for i1 in range(N1):
                            for j1 in range(N1):
                                if (i1 > j1):
                                    continue
                                B_tmp[j,i1, j1] = (BBridge_val[i1] - BBridge_val[j1])**2
                    values[i] = B_tmp.mean(axis = (1,2)).sum()
            else:
                values = np.zeros(d)
                for i in range(d):
                    B_tmp = np.zeros((self.no,N1, N1))
                    for j in range(self.no):
                        for i1 in range(N1):
                            for j1 in range(N1):
                                if (i1 > j1):
                                    continue
                                B_tmp[j,i1, j1] = (self.BBridges[j,i1, i] - self.BBridges[j,j1,i])**2
                    values[i] = B_tmp.mean(axis = (1,2)).sum()
            
            self.p_epidemic = np.mean(values >= self.Tn_epidemic)
            self.sim_values_epidemic = values
            
            self.dat_a_epidemic = np.column_stack((self.f[:,0:self.tau1],  self.f[:,self.tau2:]))
            self.dat_b_epidemic = self.f[:,self.tau1:self.tau2]
            self.warp_a_epidemic = np.column_stack((self.warp_data.gam[:,0:self.tau1],  self.warp_data.gam[:,self.tau2:]))
            self.warp_b_epidemic = self.warp_data.gam[:,self.tau1:self.tau2]
            self.mean_a_epidemic = np.column_stack((self.warp_data.fn[:,0:self.tau1],  self.warp_data.fn[:,self.tau2:])).mean(axis=1)
            self.mean_b_epidemic = self.warp_data.fn[:,self.tau1:self.tau2].mean(axis=1)  
            if self.warp_a_epidemic.size == 0:
                self.warp_mean_a_epidemic = np.array([])
            else:
                mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_a_epidemic)
                self.warp_mean_a_epidemic = gam_mu
            
            if self.warp_b_epidemic.size == 0:
                self.warp_mean_b_epidemic = np.array([])
            else:
                mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_b_epidemic)
                self.warp_mean_b_epidemic = gam_mu
        
            self.delta_epidemic = self.mean_b_epidemic - self.mean_a_epidemic
            if self.warp_mean_a_epidemic.size > 0 and self.warp_mean_b_epidemic.size > 0:
                self.delta_warp_epidemic = self.warp_mean_b_epidemic - self.warp_mean_a_epidemic
            else:
                self.delta_warp_epidemic = self.time
        else:
            self.compute_epidemic = False
            
        return


    def plot(self):
        """
        plot elastic changepoint results
        
        Usage: obj.plot()
        """

        fig, ax = plt.subplots()
        ax.plot(self.time, self.f, '0.8')
        ax.plot(self.time, self.dat_a, 'pink')
        ax.plot(self.time, self.dat_b, 'lightskyblue')
        ax.plot(self.time, self.mean_a, 'red', label='Before')
        ax.plot(self.time, self.mean_b, 'blue', label='After')
        plt.title('Functional Data')
        plt.legend()

        fig, ax = plt.subplots()
        ax.plot(self.time, self.delta, 'black')
        plt.title('Estimated Amplitude Change Function')
       
        fig, ax = plt.subplots()
        ax.plot(self.time, self.warp_data.gam, '0.8')
        ax.plot(self.time, self.warp_a, 'pink')
        ax.plot(self.time, self.warp_b, 'lightskyblue')
        ax.plot(self.time, self.warp_mean_a, 'red', label='Before')
        ax.plot(self.time, self.warp_mean_b, 'blue', label='After')
        ax.set_aspect('equal')
        plt.title('Warping Functions')
        plt.legend()

        fig, ax = plt.subplots()
        ax.plot(self.time, self.delta_warp, 'black')
        plt.title('Estimated Phase Change Function')

        plt.show()
        
        if self.compute_epidemic:
            fig, ax = plt.subplots()
            ax.plot(self.time, self.f, '0.8')
            ax.plot(self.time, self.dat_a_epidemic, 'pink')
            ax.plot(self.time, self.dat_b_epidemic, 'lightskyblue')
            ax.plot(self.time, self.mean_a_epidemic, 'red', label='Before/After')
            ax.plot(self.time, self.mean_b_epidemic, 'blue', label='During')
            plt.title('Functional Data, Epidemic')
            plt.legend()

            fig, ax = plt.subplots()
            ax.plot(self.time, self.delta_epidemic, 'black')
            plt.title('Estimated Amplitude Change Function, Epidemic')
       
            fig, ax = plt.subplots()
            ax.plot(self.time, self.warp_data.gam, '0.8')
            ax.plot(self.time, self.warp_a_epidemic, 'pink')
            ax.plot(self.time, self.warp_b_epidemic, 'lightskyblue')
            ax.plot(self.time, self.warp_mean_a, 'red', label='Before/After')
            ax.plot(self.time, self.warp_mean_b, 'blue', label='During')
            ax.set_aspect('equal')
            plt.title('Warping Functions')
            plt.legend()

            fig, ax = plt.subplots()
            ax.plot(self.time, self.delta_warp_epidemic, 'black')
            plt.title('Estimated Phase Change Function, Epidemic')

            plt.show()
            
            fig, ax = plt.subplots()
            ax.imshow(self.Sn_epidemic)
            plt.title('Test statistic, Epidemic')
        
            plt.show()

        return


class elastic_amp_change_ff:
    """"
    This class provides elastic changepoint using elastic FDA.
    It is fully-functional and an extension of the methodology of Aue et al.

    Usage:  obj = elastic_amp_change_ff(f,time)
    
    :param f: (M,N) % matrix defining N functions of M samples
    :param time: time vector of length M
    :param warp_data: aligned data (default: None)
    :param Sn: test statistic values
    :param Tn: max of test statistic 
    :param p: p-value
    :param k_star: change point
    :param values: values of computed Brownian Bridges
    :param dat_a: data before changepoint
    :param dat_b: data after changepoint
    :param warp_a: warping functions before changepoint
    :param warp_b: warping functions after changepoint
    :param mean_a: mean function before changepoint
    :param mean_b: mean function after changepoint
    :param warp_mean_a: mean warping function before changepoint
    :param warp_mean_b: mean warping function after changepoint

    Author :  J. Derek Tucker <jdtuck AT sandia.gov> and Drew Yarger <anyarge AT sandia.gov>
    Date   :  24-Aug-2022

    """
    
    def __init__(self, f, time, smooth_data=False, sparam =25, use_warp_data=False, warp_data=None, parallel=False):
        """
        Construct an instance of the elastic_change class
        :param f: (M,N) % matrix defining N functions of M samples
        :param time: time vector of length M
        :param smooth_data: smooth function data (default: False)
        :param sparam: number of smoothing runs of box filter (default: 25)
        :param use_warp_data: use precomputed warping data (default: False)
        :param warp_data: precomputed aligned data (default: None)
        :param parallel: run computation in parallel (default: True)
        """
        self.f = f
        self.time = time

        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        if use_warp_data: 
            self.warp_data = warp_data  # Align Data
        else:
            self.warp_data = fs.fdawarp(self.f,self.time)
            self.warp_data.srsf_align(parallel=parallel)
        
        
    def compute(self, d=1000, h=0, M_approx = 365, compute_epidemic = False):
        """
        Compute elastic change detection
        :param d: number of monte carlo iterations to compute p-value
        :param h: index of window type to compute long run covariance
        :param M_approx: number of time points to compute p-value
        :param compute_epidemic: compute epidemic changepoint model (default: False)
        """

        M = self.f.shape[0]
        N1 = self.f.shape[1]
        
        self.mu = np.zeros((M, N1))
        self.mu_f = np.zeros((M, N1))
        
        # Compute Karcher mean for first i+1 functions
        self.mu[:,0] = self.warp_data.qn[:,0]
        self.mu_f[:,0] = self.warp_data.fn[:,0]
        for i in range(1, N1):
            self.mu[:,i] = self.warp_data.qn[:,0:(i+1)].sum(axis = 1)
            self.mu_f[:,i] = self.warp_data.fn[:,0:(i+1)].sum(axis = 1)
        
        # compute test statistic
        self.Sn = np.zeros(N1+1)
        for k in range(1, N1+1):
            self.Sn[k] = 1/M * norm(1/np.sqrt(N1)*(self.mu_f[:,k-1] - (k/N1)*self.mu_f[:,-1]))**2
        
        self.k_star = np.argmax(self.Sn)
        self.Tn = np.max(self.Sn)
        
        # compute means on either side of the changepoint
        self.dat_a = self.f[:,range(self.k_star)]
        self.dat_b = self.f[:,range(self.k_star,N1)]
        self.warp_a = self.warp_data.gam[:,0:self.k_star]
        self.warp_b = self.warp_data.gam[:,self.k_star:]
        self.mean_a = self.warp_data.fn[:,0:self.k_star].mean(axis=1)
        self.mean_b = self.warp_data.fn[:,self.k_star:].mean(axis=1)

        if self.warp_a.size == 0:
            self.warp_mean_a = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_a)
            self.warp_mean_a = gam_mu
            
        if self.warp_b.size == 0:
            self.warp_mean_b = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_b)
            self.warp_mean_b = gam_mu
        
        self.delta = self.mean_b - self.mean_a
        if self.warp_mean_a.size > 0 and self.warp_mean_b.size > 0:
            self.delta_warp = self.warp_mean_b - self.warp_mean_a
        else:
            self.delta_warp = self.time

        # center your data
        self.centered_data = np.zeros((M, N1))
        for i in range(self.k_star):
            self.centered_data[:,i] = self.warp_data.fn[:,i] - self.mean_a
        for i in range(self.k_star, N1):
            self.centered_data[:,i] = self.warp_data.fn[:,i] - self.mean_b

        # estimate eigenvalues of covariance operator
        D_mat = LongRunCovMatrixPrecentered(self.centered_data, h = h)
        self.lambda_vals,  self.pca_coef = np.linalg.eig(D_mat)
        self.lambda_vals =   self.lambda_vals.real/M
        self.pca_coef =   self.pca_coef.real

        def asymp(N):
            BridgeLam = np.zeros((M,N))
            for j in range(M):
                BridgeLam[j,:]=self.lambda_vals[j]*(BBridge(0,0,0,1,N-1)**2)
            return max(BridgeLam.sum(axis=0))

        values = np.zeros(d)
        for sim in range(d): 
            values[sim] = asymp(N1)

        z = self.Tn <= values
        self.p = z.mean()
        self.values = values
        
        if compute_epidemic:   
            self.compute_epidemic = True
            snk_minus_vals = np.zeros((N1, N1, M))
            snk_minus_summary = np.zeros((N1, N1))
            for i in range(N1):
                for j in range(N1):
                    if (i > j):
                        continue
                    snk_minus_vals[i,j,:] = 1/np.sqrt(N1)*(self.warp_data.fn[:,i:(j+1)].sum(axis = 1) - ((j - i + 1)/N1)*self.mu_f[:,-1])
                    snk_minus_summary[i,j] = 1/M * (norm(snk_minus_vals[i,j,:])**2)
            self.tau1 = np.argmax(snk_minus_summary.max(axis = 1))
            self.tau2 = np.argmax(snk_minus_summary[self.tau1,:])
            self.Sn_epidemic = snk_minus_summary
            self.Tn_epidemic = snk_minus_summary.max()
            
            # compute means on either side of the changepoint
            self.dat_a_epidemic = self.f[:,range(self.tau1, self.tau2+1)]
            self.dat_b_epidemic = np.column_stack((self.f[:,range(0,self.tau1)],self.f[:,range(self.tau2+1,N1)]))
            self.warp_a_epidemic = self.warp_data.gam[:,range(self.tau1, self.tau2+1)]
            self.warp_b_epidemic =  np.column_stack((self.warp_data.gam[:,range(0,self.tau1)],self.warp_data.gam[:,range(self.tau2+1,N1)]))
            self.mean_a_epidemic = self.warp_data.fn[:,range(self.tau1, self.tau2+1)].mean(axis=1)
            self.mean_b_epidemic =  np.column_stack((self.warp_data.fn[:,range(0,self.tau1)],self.warp_data.fn[:,range(self.tau2+1,N1)])).mean(axis = 1)
            
            
            if self.warp_a_epidemic.size == 0:
                self.warp_mean_a_epidemic = np.array([])
            else:
                mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_a_epidemic)
                self.warp_mean_a_epidemic = gam_mu
            
            if self.warp_b_epidemic.size == 0:
                self.warp_mean_b_epidemic = np.array([])
            else:
                mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_b_epidemic)
                self.warp_mean_b_epidemic = gam_mu
        
            self.delta_epidemic = self.mean_b_epidemic - self.mean_a_epidemic
            if self.warp_mean_a_epidemic.size > 0 and self.warp_mean_b_epidemic.size > 0:
                self.delta_warp_epidemic = self.warp_mean_b_epidemic - self.warp_mean_a_epidemic
            else:
                self.delta_warp_epidemic = self.time

            # center your data
            self.centered_data_epidemic = np.zeros((M, N1))
            for i in range(N1):
                if i in range(self.tau1, self.tau2+1):
                    self.centered_data_epidemic[:,i] = self.warp_data.fn[:,i] - self.mean_b
                else:
                    self.centered_data_epidemic[:,i] = self.warp_data.fn[:,i] - self.mean_a
                    # estimate eigenvalues of covariance operator
            D_mat = LongRunCovMatrixPrecentered(self.centered_data_epidemic, h = h)
            self.lambda_vals_epidemic,  self.pca_coef_epidemic = np.linalg.eig(D_mat)
            self.lambda_vals_epidemic =   self.lambda_vals_epidemic.real / M
            self.pca_coef_epidemic =   self.pca_coef_epidemic.real * M

            def asymp(N):
                BridgeLam = np.zeros((M_approx, N, N))
                for j in range(M_approx):
                    BB = BBridge(0,0,0,1,N-1)
                    BB_sub = np.broadcast_to(BB, (N, N)) -  np.broadcast_to(BB, (N, N)).T
                    BridgeLam[j,:,:]=self.lambda_vals_epidemic[j]*(BB_sub**2)
                return BridgeLam.sum(axis=0).max()

            values = np.zeros(d)
            for sim in range(d): 
                values[sim] = asymp(N1)

            z = self.Tn_epidemic <= values
            self.p_epidemic = z.mean()
            self.values_epidemic = values
        else:
            self.compute_epidemic = False
        
        return
    
    def plot(self):
        """
        plot elastic changepoint results
        
        Usage: obj.plot()
        """

        fig, ax = plt.subplots()
        ax.plot(self.time, self.f, '0.8')
        ax.plot(self.time, self.dat_a, 'pink')
        ax.plot(self.time, self.dat_b, 'lightskyblue')
        ax.plot(self.time, self.mean_a, 'red', label='Before')
        ax.plot(self.time, self.mean_b, 'blue', label='After')
        plt.title('Functional Data')
        plt.legend()

        fig, ax = plt.subplots()
        ax.plot(self.time, self.delta, 'black')
        plt.title('Estimated Amplitude Change Function')
       
        fig, ax = plt.subplots()
        ax.plot(self.time, self.warp_data.gam, '0.8')
        ax.plot(self.time, self.warp_data.gam[:,range(self.k_star)], 'pink')
        ax.plot(self.time, self.warp_data.gam[:,self.k_star:], 'lightskyblue')
        ax.plot(self.time, self.warp_mean_a, 'red', label='Before')
        ax.plot(self.time, self.warp_mean_b, 'blue', label='After')
        ax.set_aspect('equal')
        plt.title('Warping Functions')
        plt.legend()
        
        fig, ax = plt.subplots()
        ax.plot(list(range(self.Sn.shape[0])), self.Sn)
        plt.title('Test statistic')

        plt.show()
        
                
        if self.compute_epidemic:
            fig, ax = plt.subplots()
            ax.plot(self.time, self.f, '0.8')
            ax.plot(self.time, self.dat_a_epidemic, 'pink')
            ax.plot(self.time, self.dat_b_epidemic, 'lightskyblue')
            ax.plot(self.time, self.mean_a_epidemic, 'red', label='Before/After')
            ax.plot(self.time, self.mean_b_epidemic, 'blue', label='During')
            plt.title('Functional Data, Epidemic')
            plt.legend()

            fig, ax = plt.subplots()
            ax.plot(self.time, self.delta_epidemic, 'black')
            plt.title('Estimated Amplitude Change Function, Epidemic')
       
            fig, ax = plt.subplots()
            ax.plot(self.time, self.warp_data.gam, '0.8')
            ax.plot(self.time, self.warp_a_epidemic, 'pink')
            ax.plot(self.time, self.warp_b_epidemic, 'lightskyblue')
            ax.plot(self.time, self.warp_mean_a, 'red', label='Before/After')
            ax.plot(self.time, self.warp_mean_b, 'blue', label='During')
            ax.set_aspect('equal')
            plt.title('Warping Functions')
            plt.legend()

            fig, ax = plt.subplots()
            ax.plot(self.time, self.delta_warp_epidemic, 'black')
            plt.title('Estimated Phase Change Function, Epidemic')

            plt.show()
            
            fig, ax = plt.subplots()
            ax.imshow(self.Sn_epidemic)
            plt.title('Test statistic, Epidemic')
        
            plt.show()


        return


class elastic_ph_change_ff:
    """"
    This class provides elastic changepoint using elastic FDA on warping functions.
    It is fully-functional and an extension of the methodology of Aue et al.

    Usage:  obj = elastic_ph_change_ff(f,time)
    
    :param f: (M,N) % matrix defining N functions of M samples
    :param time: time vector of length M
    :param warp_data: aligned data (default: None)
    :param Sn: test statistic values
    :param Tn: max of test statistic 
    :param p: p-value
    :param k_star: change point
    :param values: values of computed Brownian Bridges
    :param dat_a: data before changepoint
    :param dat_b: data after changepoint
    :param warp_a: warping functions before changepoint
    :param warp_b: warping functions after changepoint
    :param mean_a: mean function before changepoint
    :param mean_b: mean function after changepoint
    :param warp_mean_a: mean warping function before changepoint
    :param warp_mean_b: mean warping function after changepoint

    Author :  J. Derek Tucker <jdtuck AT sandia.gov>
    Date   :  17-Nov-2022

    """
    
    def __init__(self, f, time, smooth_data=False, sparam=25, use_warp_data=False, warp_data=None, parallel=False):
        """
        Construct an instance of the elastic_change class
        :param f: (M,N) % matrix defining N functions of M samples
        :param time: time vector of length M
        :param smooth_data: smooth function data (default: False)
        :param sparam: number of smoothing runs of box filter (default: 25)
        :param use_warp_data: use precomputed warping data (default: False)
        :param warp_data: precomputed aligned data (default: None)
        :param parallel: run computation in parallel (default: True)
        """
        
        self.f = f
        self.time = time

        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        
        if use_warp_data: 
            self.warp_data = warp_data# Align Data
        else:
            self.warp_data = fs.fdawarp(self.f,self.time)
            self.warp_data.srsf_align(parallel=parallel)
        

    def compute(self, d=1000, h=0, M_approx = 365):
        """
        Compute elastic change detection
        :param d: number of monte carlo iterations to compute p-value
        :param h: index of window type to compute long run covariance
        :param M_approx: number of time points to compute p-value
        """

        M = self.f.shape[0]
        N1 = self.f.shape[1]
        
        self.mu = np.zeros((M, N1))
        
        # Compute Karcher mean of warping functions
        mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_data.gam)
        self.vec = vec

        self.mu[:,0] = vec[:,0]
        for i in range(1, N1):
            self.mu[:,i] = vec[:,0:(i+1)].sum(axis = 1)
        
        # compute test statistic
        self.Sn = np.zeros(N1+1)
        # k indexes subscript of test statistic
        for k in range(1, N1+1):
            self.Sn[k] = norm(1/np.sqrt(N1)*(self.mu[:,k-1] - (k/N1)*self.mu[:,-1]))**2 / M

        self.k_star = np.argmax(self.Sn)
        self.Tn = np.max(self.Sn)
        
        # compute means on either side of the changepoint
        self.dat_a = self.f[:,range(self.k_star)]
        self.dat_b = self.f[:,range(self.k_star,N1)]
        self.warp_a = self.warp_data.gam[:,0:self.k_star]
        self.warp_b = self.warp_data.gam[:,self.k_star:]
        self.mean_a = self.warp_data.fn[:,0:self.k_star].mean(axis=1)
        self.mean_b = self.warp_data.fn[:,self.k_star:].mean(axis=1)

        if self.warp_a.size == 0:
            mu_a = np.zeros(M)
            self.warp_mean_a = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_a)
            mu_a = vec.mean(axis=1)
            self.warp_mean_a = gam_mu
            
        if self.warp_b.size == 0:
            mu_b = np.zeros(M)
            self.warp_mean_b = np.array([])
        else:
            mu, gam_mu, psi, vec = uf.SqrtMean(self.warp_b)
            mu_b = vec.mean(axis=1)
            self.warp_mean_b = gam_mu
        
        self.delta = self.mean_b - self.mean_a
        if self.warp_mean_a.size > 0 and self.warp_mean_b.size > 0:
            self.delta_warp = self.warp_mean_b - self.warp_mean_a
        else:
            self.delta_warp = self.time

        # center your data
        self.centered_data = np.zeros((M, N1))
        for i in range(self.k_star):
            self.centered_data[:,i] = self.vec[:,i] - mu_a
        for i in range(self.k_star, N1):
            self.centered_data[:,i] = self.vec[:,i] - mu_b

        # estimate eigenvalues of covariance operator
        D_mat = LongRunCovMatrixPrecentered(self.centered_data, h = h)
        self.lambda_vals,  self.pca_coef = np.linalg.eig(D_mat)
        self.lambda_vals =   self.lambda_vals.real / M
        self.pca_coef =   self.pca_coef.real * M

        def asymp(N):
            BridgeLam = np.zeros((M,N))
            for j in range(M):
                BridgeLam[j,:]=self.lambda_vals[j]*(BBridge(0,0,0,1,N-1)**2)
            return max(BridgeLam.sum(axis=0))

        values = np.zeros(d)
        for sim in range(d): 
            values[sim] = asymp(N1)

        z = self.Tn <= values
        self.p = z.mean()
        self.values = values

        return
    
    def plot(self):
        """
        plot elastic changepoint results
        
        Usage: obj.plot()
        """

        fig, ax = plt.subplots()
        ax.plot(self.time, self.f, '0.8')
        ax.plot(self.time, self.dat_a, 'pink')
        ax.plot(self.time, self.dat_b, 'lightskyblue')
        ax.plot(self.time, self.mean_a, 'red', label='Before')
        ax.plot(self.time, self.mean_b, 'blue', label='After')
        plt.title('Functional Data')
        plt.legend()

        fig, ax = plt.subplots()
        ax.plot(self.time, self.delta, 'black')
        plt.title('Estimated Amplitude Change Function')
       
        fig, ax = plt.subplots()
        ax.plot(self.time, self.warp_data.gam, '0.8')
        ax.plot(self.time, self.warp_data.gam[:,range(self.k_star)], 'pink')
        ax.plot(self.time, self.warp_data.gam[:,self.k_star:], 'lightskyblue')
        ax.plot(self.time, self.warp_mean_a, 'red', label='Before')
        ax.plot(self.time, self.warp_mean_b, 'blue', label='After')
        ax.set_aspect('equal')
        plt.title('Warping Functions')
        plt.legend()
        
        fig, ax = plt.subplots()
        ax.plot(list(range(self.Sn.shape[0])), self.Sn)
        plt.title('Test statistic')

        plt.show()

        return
