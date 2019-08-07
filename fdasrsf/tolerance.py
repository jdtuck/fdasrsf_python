"""
Functional Tolerance Bounds using SRSF

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
from fdasrsf.gauss_model import joint_gauss_model
from fdasrsf.boxplots import ampbox, phbox


def bootTB(f, time, a=0.5, p=.99, B=500, no=5, parallel=True):
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
    :return ph: phase tolerance bounds

    """

    (M, N) = f.shape

    # Align Data
    out_med = fs.srsf_align(f, time, method="median", showplot=False, parallel=parallel)

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
        samples = joint_gauss_model(fn, time, qn, gam, q0, n=100, no=no)
        obja = ampbox(samples.ft, out_med.fmean , samples.qs, out_med.mqn, time, alpha=1-p, k_a=.3)
        objp = phbox(samples.gams, time, alpha=1-p, k_a=.3)
        bootlwr_amp[:,k] = obja.Q1a
        bootupr_amp[:,k] = obja.Q3a
        bootlwr_ph[:,k] = objp.Q1a
        bootupr_ph[:,k] = objp.Q3a
    
    # tolerance bounds
    boot_amp = np.hstack((bootlwr_amp, bootupr_amp)
    boot_amp_q = uf.f_to_srsf(boot_amp,time)
    boot_ph = np.hstack((bootlwr_ph,bootupr_ph))
    amp = ampbox(boot_amp, out_med.fmean , boot_amp_q, out_med.mqn, time, alpha=a, k_a=.3)
    ph = phbox(boot_ph, time, alpha=a, k_a=.3)

    return amp, ph
    

    