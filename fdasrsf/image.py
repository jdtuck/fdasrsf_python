"""
image warping using SRVF framework

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf.image_functions as fif
import cimage as im


def reparam_image(It, Im, gam=None, b=None, stepsize=1e-4, itermax=20):
    """
    This function warps an image to another using SRVF framework

    :param Im: numpy ndarray of shape (N,N) representing a NxN image
    :param Im: numpy ndarray of shape (N,N) representing a NxN image
    :param gam: numpy ndarray of shape (N,N) representing an initial warping function
    :param b: numpy ndarray representing basis matrix

    :rtype: numpy ndarray
    :return gamnew: diffeomorphism
    :return Inew: warped image
    :return H: energy
    :return stepsize: final stepsize

    """

    m = It.shape[0]
    n = It.shape[1]

    gamid = fif.makediffeoid(m,n)

    if gam is None:
        gam = gamid.copy()
    
    if b is None: 
        M = 10
        basetype = 't'
        b = fif.formbasisTid(M, m, n, basetype)

    # main loop
    H = np.zeros(itermax+1)
    Iold = fif.apply_gam_imag(Im, gam)
    Iold -= Iold.min()
    Iold /= Iold.max()
    qt = fif.image_to_q(It)
    qm = fif.image_to_q(Iold)
    gamold = gam.copy()

    gamnew = gamold.copy()
    Inew = Iold.copy()
    iter = 0
    H[iter] = fif.compEnergy(qt,qm)

    print('Iteration %d, energy %f\n'% (iter,H[iter]))

    gamupdate = fif.updateGam(qt,qm,b)

    cutoff = 1e-3

    for iter in range(1,itermax+1):
        gaminc = gamid + stepsize*gamupdate

        G = im.check_crossing(gamnew)

        if G == 0:
            print('Possible crossing!\n')
            gamnew = gamold.copy()
            stepsize *= 0.67
            H[iter] = H[iter-1]
            continue
        
        else:
            gamnew = fif.apply_gam_gamid(gamnew,gaminc)
        
        Inew = fif.apply_gam_imag(Im, gamnew)
        Inew -= Inew.min()
        Inew /= Inew.max()
        qm = fif.image_to_q(Inew)
        H[iter] = fif.compEnergy(qt,qm)
        print('Iteration %d, energy %f\n'% (iter,H[iter]))

        if (iter > 4):
            hstop = 1
            for i in range(4):
                hstop *= (H[iter] >= H[iter-i])
            
            if hstop != 0:
                print('Warning: energy constantly increasing\n')
                break
        
        if (iter > 4) and (np.abs(H[iter] <= H[iter-1])<cutoff and np.abs(H[iter-1] - H[iter-2])<cutoff and np.abs(H[iter-2] <= H[iter-3])<cutoff):
            print('Warning: energy is not changing\n')
            break

        if ((iter > 1) and (H[iter] > H[iter-1])) or ((iter > 3) and ((H[iter-1] <= H[iter-2]) and (H[iter-2] > H[iter-3]))):
            stepsize *= 0.9
            gamnew = gamold.copy()
            H[iter] = H[iter-1]
            continue
        
        gamold = gamnew.copy()
        gamupdate = fif.updateGam(qt,qm,b)
    
    H = H[0:iter]

    return gamnew,Inew,H,stepsize