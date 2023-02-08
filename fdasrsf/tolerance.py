"""
Functional Tolerance Bounds using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
import fdasrsf.fPCA as fpca
from scipy.stats import chi2
from numpy.linalg import eig
from fdasrsf.boxplots import ampbox, phbox


def bootTB(f, time, a=0.05, p=.99, B=500, no=5, parallel=True):
    """
    This function computes tolerance bounds for functional data containing
    phase and amplitude variation using bootstrap sampling

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param a: confidence level of tolerance bound (default = 0.05)
    :param p: coverage level of tolerance bound (default = 0.99)
    :param B: number of bootstrap samples (default = 500)
    :param no: number of principal components (default = 5)
    :param parallel: enable parallel processing (default = T)
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of boxplot objects
    :return amp: amplitude tolerance bounds
    :rtype out_med: ampbox object
    :return ph: phase tolerance bounds
    :rtype out_med: phbox object
    :return out_med: alignment results
    :rtype out_med: fdawarp object

    """
    eps = np.finfo(np.double).eps
    (M, N) = f.shape

    # Align Data
    out_med = fs.fdawarp(f, time)
    out_med.srsf_align(method="median", parallel=parallel)

    # Calculate CI
    # a% tolerance bound with p%
    fn = out_med.fn
    qn = out_med.qn
    gam = out_med.gam
    q0 = out_med.q0
    print("Bootstrap Sampling")
    bootlwr_amp = np.zeros((M,B))
    bootupr_amp = np.zeros((M,B))
    bootlwr_ph =  np.zeros((M,B))
    bootupr_ph =  np.zeros((M,B))
    for k in range(B):
        out_med.joint_gauss_model(n=100, no=no)
        obja = ampbox(out_med)
        obja.construct_boxplot(1-p,.3)
        objp = phbox(out_med)
        objp.construct_boxplot(1-p,.3)
        bootlwr_amp[:,k] = obja.Q1a
        bootupr_amp[:,k] = obja.Q3a
        bootlwr_ph[:,k] = objp.Q1a
        bootupr_ph[:,k] = objp.Q3a
    
    # tolerance bounds
    boot_amp = np.hstack((bootlwr_amp, bootupr_amp))
    f, g, g2 = uf.gradient_spline(time, boot_amp, False)
    boot_amp_q = g / np.sqrt(abs(g) + eps)
    boot_ph = np.hstack((bootlwr_ph,bootupr_ph))
    boot_out = out_med
    boot_out.fn = boot_amp
    boot_out.qn = boot_amp_q
    boot_out.gam = boot_ph
    boot_out.rsamps = False
    amp = ampbox(boot_out)
    amp.construct_boxplot(a, .3)
    ph = phbox(boot_out)
    ph.construct_boxplot(a, .3)

    return amp, ph, out_med


def pcaTB(f, time, a=0.5, p=.99, no=5, parallel=True):
    """
    This function computes tolerance bounds for functional data containing
    phase and amplitude variation using fPCA

    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param time: vector of size M describing the sample points
    :param a: confidence level of tolerance bound (default = 0.05)
    :param p: coverage level of tolerance bound (default = 0.99)
    :param no: number of principal components (default = 5)
    :param parallel: enable parallel processing (default = T)
    :type f: np.ndarray
    :type time: np.ndarray

    :rtype: tuple of boxplot objects
    :return warp: alignment data from time_warping
    :return pca: functional pca from jointFPCA
    :return tol: tolerance factor

    """

    # Align Data
    out_warp = fs.fdawarp(f, time)
    out_warp.srsf_align( method="median", parallel=parallel)

    # Calculate pca
    out_pca = fpca.fdajpca(out_warp)
    out_pca.calc_fpca(no)

    # Calculate TB
    tol = mvtol_region(out_pca.coef, a, p, 100000)

    return warp, pca, tol


def mvtol_region(x, alpha, P, B):
    """
    Computes tolerance factor for multivariate normal

    Krishnamoorthy, K. and Mondal, S. (2006), Improved Tolerance Factors for Multivariate Normal
    Distributions, Communications in Statistics - Simulation and Computation, 35, 461–478.
    
    :param x: (M,N) matrix defining N variables of M samples
    :param alpha: confidence level
    :param P: coverage level
    :param B: number of bootstrap samples

    :rtype: double
    :return tol: tolerance factor

    """
    n,p = x.shape

    q_squared = chi2.rvs(1, size=(p,B))/n
    L = np.zeros((p,B))
    for k in range(B):
        L[:,k] = eig(rwishart(n-1,p))[0]
    
    c1 = (1+q_squared)/L
    c1 = c1.sum()
    c2 = (1+2*q_squared)/(L**2)
    c2 = c2.sum()
    c3 = (1+3*q_squared)/(L**3)
    c3 = c3.sum()
    a = (c2**3)/(c3**2)
    T = (n-1)*(np.sqrt(c2/a) * (chi2.ppf(P,a)-a) + c1)
    tol = np.quantile(T,1-alpha)

    return tol


def rwishart(df,p):
    """
    Computes a random wishart matrix
    
    :param df: degree of freedom
    :param p: number of dimensions

    :rtype: double
    :return R: matrix

    """
    R = np.zeros((p,p))
    R = R.flatten()
    R[::p+1] = np.sqrt(chi2.rvs(np.arange(df,df-p,-1),size=p))
    if p>1:
        pseq = np.arange(0,p)
        tmp = [np.arange(0,x+1) for x in np.arange(0,p-1)]
        R[np.repeat(p*pseq,pseq)+np.concatenate(tmp).ravel()] = np.random.randn(int(p*(p-1)/2))

    R = R.reshape((p,p))
    R = np.matmul(R.T,R)

    return R