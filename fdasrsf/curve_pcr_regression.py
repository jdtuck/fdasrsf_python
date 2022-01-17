"""
Warping Invariant PCR Regression using SRSF

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
import fdasrsf.curve_functions as cf
import fdasrsf.regression as rg
import fdasrsf.geometry as geo
from scipy.linalg import inv, norm
from scipy.integrate import trapz, cumtrapz

class elastic_curve_pcr_regression:
    """
    This class provides elastic curve pcr regression for functional data using the
    SRVF framework accounting for warping
    
    Usage:  obj = elastic_curve_pcr_regression(beta,y)
    
    :param beta: numpy ndarray of shape (n, M, N) describing N curves in R^M
    :param y: response vector of length N
    :param warp_data: fdacurve object of alignment and shape PCA
    :param alpha: intercept
    :param b: coefficient vector
    :param SSE: sum of squared errors

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  29-Oct-2021
    """

    def __init__(self, beta, y):
        """
        Construct an instance of the elastic_curve_pcr_regression class
        :param beta: numpy ndarray of shape (n, M, N) describing N curves in R^M
        :param y: response vector
        :param time: vector of size M describing the sample points
        """

        self.beta = beta
        self.y = y

    def calc_model(self, no=5, T=None, rotation=True, parallel=False):
        """
        This function identifies a regression model with phase-variability
        using elastic pca

        :param no: scalar specify number of principal components (default=5)
        :param T: number of resample curve to 
        :param rotation: include rotation (default = T)
        :param parallel: run in parallel (default = F)
        """
        
        self.rotation = rotation
        M = self.beta.shape[1]
        if T is None:
            T = M

        # Align Data
        self.warp_data = fs.fdacurve(self.beta, N=T)
        self.warp_data.karcher_mean(rotation=rotation, parallel=parallel)

        # Calculate PCA
        self.warp_data.shape_pca(no=no)
        
        # OLS using PCA basis
        lam = 0
        R = 0
        N1 = self.warp_data.coef.shape[1]
        Phi = np.ones((N1, no+1))
        Phi[:,1:(no+1)] = self.warp_data.coef.T
        xx = np.dot(Phi.T, Phi)
        inv_xx = inv(xx + lam * R)
        xy = np.dot(Phi.T, self.y)
        b = np.dot(inv_xx, xy)
        alpha = b[0]
        b = b[1:no+1]

        # compute the SSE
        int_X = np.zeros(N1)
        for ii in range(0,N1):
            int_X[ii] = np.sum(self.warp_data.coef[:,ii]*b)
        
        SSE = np.sum((self.y-alpha-int_X)**2)

        self.alpha = alpha
        self.b = b
        self.SSE = SSE

        return
    

    def predict(self, newdata=None):
        """
        This function performs prediction on regression model on new data if available or current stored data in object
        Usage:  obj.predict()
                obj.predict(newdata)

        :param newdata: dict containing new data for prediction (needs the keys below, if None predicts on training data)
        :type newdata: dict
        :param beta: (n, M,N) matrix of curves
        :param y: truth if available
        """

        T = self.warp_data.beta_mean.shape[1]
        if newdata != None:
            beta = newdata['beta']
            y = newdata['y']
            n = beta.shape[2]
            beta1 = np.zeros(beta.shape)
            q = np.zeros(beta.shape)
            for ii in range(0,n):
                if (beta.shape[1] != T):
                    beta1[:,:,ii] = cf.resamplecurve(beta[:,:,ii],T)
                else:
                    beta1[:,:,ii] = beta[:,:,ii]
                a = -cf.calculatecentroid(beta1[:,:,ii])
                beta1[:,:,ii] += np.tile(a, (T,1)).T
                q[:,:,ii] = cf.curve_to_q(beta1[:,:,ii])[0]
            
            mu = self.warp_data.q_mean

            v = np.zeros(q.shape)
            for ii in range(0,n):
                qn_t, R, gamI = cf.find_rotation_and_seed_unique(mu, q[:,:,ii], 0, self.rotation)
                qn_t = qn_t / np.sqrt(cf.innerprod_q2(qn_t,qn_t))

                q1dotq2 = cf.innerprod_q2(mu,qn_t)

                if (q1dotq2 > 1):
                    q1dotq2 = 1
                
                d = np.arccos(q1dotq2)

                u = qn_t - q1dotq2*mu
                normu = np.sqrt(cf.innerprod_q2(u,u))
                if (normu>1e-4):
                    v[:,:,ii] = u*np.arccos(q1dotq2)/normu
                else:
                    v[:,:,ii] = np.zeros(qn_t.shape)

            
            Utmp = self.warp_data.U.T
            no = self.warp_data.U.shape[1]
            VM = np.mean(self.warp_data.v,2)
            VM = VM.flatten()

            x = np.zeros((no,n))
            for i in range(0,n):
                tmp = v[:,:,i]
                tmpv1 = tmp.flatten()
                x[:,i] = Utmp.dot((tmpv1- VM))

            self.y_pred = np.zeros(n)
            for ii in range(0,n):
                self.y_pred[ii] = self.alpha + np.dot(x[:,ii],self.b)
            
            if y is None:
                self.SSE = np.nan
            else:
                self.SSE = np.sum((y-self.y_pred)**2)
        else:
            n = self.warp_data.coef.shape[1]
            self.y_pred = np.zeros(n)
            for ii in range(0,n):
                self.y_pred[ii] = self.alpha + np.dot(self.warp_data.coef[:,ii],self.b)
            
            self.SSE = np.sum((self.y-self.y_pred)**2)

        return
