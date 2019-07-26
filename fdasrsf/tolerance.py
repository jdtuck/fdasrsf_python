"""
Functional Tolerance Bounds using SRSF

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf

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

    :rtype: tuple of numpy array
    :return amp: amplitude tolerance bounds
    :return ph: phase tolerance bounds

    """

    (M, N) = f.shape

    # Align Data
    out_med = fs.srsf_align(f, time, method="median", showplot=False, parallel=parallel)

    # Calculate CI
    # a% tolerance bound with p%
    print("Bootstrap Sampling")
    k = 1
    