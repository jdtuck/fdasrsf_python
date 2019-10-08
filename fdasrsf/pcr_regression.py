"""
Warping PCR Invariant Regression using SRSF

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
import fdasrsf.fPCA as fpca
import fdasrsf.regression as rg
from scipy import dot
from scipy.linalg import inv, norm
from scipy.integrate import trapz, cumtrapz
from scipy.optimize import fmin_l_bfgs_b
import collections

class elastic_pcr_regression:
    """
    This class provides elastic pcr regression for functional data using the
    SRVF framework accounting for warping
    
    Usage:  obj = elastic_pcr_regression(f,y,time)
    
    :param f: (M,N) % matrix defining N functions of M samples
    :param y: response vector of length N
    :param warp_data: fdawarp object of alignment
    :param pca: class dependent on fPCA method used object of fPCA
    :param information
    :param alpha: intercept
    :param b: coefficient vector
    :param SSE: sum of squared errors

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  18-Mar-2018
    """

    def __init__(self, f, y, time):
        """
        Construct an instance of the elastic_pcr_regression class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param y: response vector
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.y = y
        self.time = time

    def calc_model(self, pca_method="combined", no=5, 
                   smooth_data=False, sparam=25, parallel=False, 
                   C=None):
        """
        This function identifies a regression model with phase-variability
        using elastic pca

        :param pca_method: string specifing pca method (options = "combined",
                        "vert", or "horiz", default = "combined")
        :param no: scalar specify number of principal components (default=5)
        :param smooth_data: smooth data using box filter (default = F)
        :param sparam: number of times to apply box filter (default = 25)
        :param parallel: run in parallel (default = F)
        :param C: scale balance parameter for combined method (default = None)
        """

        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        N1 = self.f.shape[1]

        # Align Data
        self.warp_data = fs.fdawarp(self.f,self.time)
        self.warp_data.srsf_align(parallel=parallel)

        # Calculate PCA
        if pca_method=='combined':
            out_pca = fpca.fdajpca(self.warp_data)
        elif pca_method=='vert':
            out_pca = fpca.fdavpca(self.warp_data)
        elif pca_method=='horiz':
            out_pca = fpca.fdahpca(self.warp_data)
        else:
            raise Exception('Invalid fPCA Method')
        out_pca.calc_fpca(no)
        
        # OLS using PCA basis
        lam = 0
        R = 0
        Phi = np.ones((N1, no+1))
        Phi[:,1:(no+1)] = out_pca.coef
        xx = dot(Phi.T, Phi)
        inv_xx = inv(xx + lam * R)
        xy = dot(Phi.T, y)
        b = dot(inv_xx, xy)
        alpha = b[0]
        b = b[1:no+1]

        # compute the SSE
        int_X = np.zeros(N1)
        for ii in range(0,N1):
            int_X[ii] = np.sum(out_pca.coef*b)
        
        SSE = np.sum((y-alpha-int_X)**2)

        self.alpha = alpha
        self.b = b
        self.pca = out_pca
        self.SSE = SSE
        self.pca_method = pca_method

        return


class elastic_lpcr_regression:
    """
    This class provides elastic logistic pcr regression for functional 
    data using the SRVF framework accounting for warping
    
    Usage:  obj = elastic_lpcr_regression(f,y,time)

    :param f: (M,N) % matrix defining N functions of M samples
    :param y: response vector of length N (-1/1)
    :param warp_data: fdawarp object of alignment
    :param pca: class dependent on fPCA method used object of fPCA
    :param information
    :param alpha: intercept
    :param b: coefficient vector
    :param Loss: logistic loss
    :param PC: probability of classification
    :param ylabels: predicted labels
    
    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  18-Mar-2018
    """

    def __init__(self, f, y, time):
        """
        Construct an instance of the elastic_lpcr_regression class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param y: response vector
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.y = y
        self.time = time

    def calc_model(self, pca_method="combined", no=5, 
                   smooth_data=False, sparam=25):
        """
        This function identifies a logistic regression model with phase-variability
        using elastic pca

        :param pca_method: string specifing pca method (options = "combined",
                        "vert", or "horiz", default = "combined")
        :param no: scalar specify number of principal components (default=5)
        :param smooth_data: smooth data using box filter (default = F)
        :param sparam: number of times to apply box filter (default = 25)
        :type f: np.ndarray
        :type time: np.ndarray
        """

        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        N1 = self.f.shape[1]

        # Align Data
        self.warp_data = fs.fdawarp(self.f,self.time)
        self.warp_data.srsf_align(parallel=parallel)

        # Calculate PCA
        if pca_method=='combined':
            out_pca = fpca.fdajpca(self.warp_data)
        elif pca_method=='vert':
            out_pca = fpca.fdavpca(self.warp_data)
        elif pca_method=='horiz':
            out_pca = fpca.fdahpca(self.warp_data)
        else:
            raise Exception('Invalid fPCA Method')
        out_pca.calc_fpca(no)
        
        # OLS using PCA basis
        lam = 0
        R = 0
        Phi = np.ones((N1, no+1))
        Phi[:,1:(no+1)] = out_pca.coef
        # Find alpha and beta using l_bfgs
        b0 = np.zeros(no+1)
        out = fmin_l_bfgs_b(rg.logit_loss, b0, fprime=rg.logit_gradient,
                            args=(Phi, y), pgtol=1e-10, maxiter=200,
                            maxfun=250, factr=1e-30)

        b = out[0]
        alpha = b[0]

        # compute the Loss
        LL = rg.logit_loss(b,Phi,y)

        b = b[1:no+1]

        self.alpha = alpha
        self.b = b
        self.pca = out_pca
        self.LL = LL
        self.pca_method = pca_method

        return
        

class elastic_mlpcr_regression:
    """
    This class provides elastic multinomial logistic pcr regression for functional
    data using the SRVF framework accounting for warping
    
    Usage:  obj = elastic_mlpcr_regression(f,y,time)

    :param f: (M,N) % matrix defining N functions of M samples
    :param y: response vector of length N
    :param Y: coded label matrix
    :param warp_data: fdawarp object of alignment
    :param pca: class dependent on fPCA method used object of fPCA
    :param information
    :param alpha: intercept
    :param b: coefficient vector
    :param Loss: logistic loss
    :param PC: probability of classification
    :param ylabels: predicted labels
    :param 

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  18-Mar-2018
    """

    def __init__(self, f, y, time):
        """
        Construct an instance of the elastic_mlpcr_regression class
        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param y: response vector
        :param time: vector of size M describing the sample points
        """
        a = time.shape[0]

        if f.shape[0] != a:
            raise Exception('Columns of f and time must be equal')

        self.f = f
        self.y = y
        self.time = time

        # Code labels
        m = y.max()
        self.n_classes = m
        self.Y = np.zeros((N1, m), dtype=int)
        for ii in range(0, N1):
            self.Y[ii, y[ii]-1] = 1

    def calc_model(self, pca_method="combined", no=5, 
                   smooth_data=False, sparam=25):
        """
        This function identifies a logistic regression model with phase-variability
        using elastic pca

        :param f: numpy ndarray of shape (M,N) of N functions with M samples
        :param y: numpy array of N responses
        :param time: vector of size M describing the sample points
        :param pca_method: string specifing pca method (options = "combined",
                        "vert", or "horiz", default = "combined")
        :param no: scalar specify number of principal components (default=5)
        :param smooth_data: smooth data using box filter (default = F)
        :param sparam: number of times to apply box filter (default = 25)
        :type f: np.ndarray
        :type time: np.ndarray
        """

        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        N1 = self.f.shape[1]

        # Align Data
        self.warp_data = fs.fdawarp(self.f,self.time)
        self.warp_data.srsf_align(parallel=parallel)

        # Calculate PCA
        if pca_method=='combined':
            out_pca = fpca.fdajpca(self.warp_data)
        elif pca_method=='vert':
            out_pca = fpca.fdavpca(self.warp_data)
        elif pca_method=='horiz':
            out_pca = fpca.fdahpca(self.warp_data)
        else:
            raise Exception('Invalid fPCA Method')
        out_pca.calc_fpca(no)
        
        # OLS using PCA basis
        lam = 0
        R = 0
        Phi = np.ones((N1, no+1))
        Phi[:,1:(no+1)] = out_pca.coef
        # Find alpha and beta using l_bfgs
        b0 = np.zeros(m*(no+1))
        out = fmin_l_bfgs_b(rg.mlogit_loss, b0, fprime=rg.mlogit_gradient,
                                args=(Phi, Y), pgtol=1e-10, maxiter=200,
                                maxfun=250, factr=1e-30)

        b = out[0]
        B0 = b.reshape(no+1, m)
        alpha = B0[0, :]

        # compute the Loss
        LL = rg.mlogit_loss(b,Phi,y)

        b = B0[1:no+1,:]

        self.alpha = alpha
        self.b = b
        self.pca = out_pca
        self.LL = LL
        self.pca_method = pca_method

        return
