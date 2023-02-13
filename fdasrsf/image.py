"""
image warping using SRVF framework

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.image_functions as fif

def reparam_image(It, Im, gam=None, b=None, stepsize=1e-4, itermax=20):
    """
    This function warps an image to another using SRVF framework

    :param Im: numpy ndarray of shape (N,N) representing a NxN image
    :param Im: numpy ndarray of shape (N,N) representing a NxN image
    :param gam: numpy ndarray of shape (N,N) representing an initial warping function
    :param b: numpy ndarray representing basis matrix

    :rtype: numpy ndarray
    :return f: smoothed functions functions

    """

    m = It.shape[0]
    n = It.shape[1]

    gamid = fif.makediffeoid(m,n)

    if gam is None:
        gam = gamid.copy()

    # main loop
    H = np.zeros(itermax+1)
    Iold = fif.apply_gam_imag(Im, gam)
    Iold -= Iold.min()
    Iold /= Iold.max()
    qt = fif.image_to_q(It)
    qm = fif.image_to_q(Iold)
    gamold = gam.copy()