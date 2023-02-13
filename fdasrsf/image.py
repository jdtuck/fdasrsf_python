"""
image warping using SRVF framework

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""


def reparam_image(It, Im, gam, b, stepsize=1e-4, itermax=20):
    """
    This function warps an image to another using SRVF framework

    :param Im: numpy ndarray of shape (N,N) respresenting a NxN image
    :param Im: numpy ndarray of shape (N,N) respresenting a NxN image
    :param gam: numpy ndarray of shape (N,N) respresenting an initial warping function
    :param b: numpy ndarray of 

    :rtype: numpy ndarray
    :return f: smoothed functions functions

    """