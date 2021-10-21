"""
Warping Invariant GML Regression using SRSF

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import fdasrsf as fs
import fdasrsf.utility_functions as uf
from patsy import bs
from scipy.optimize import minimize
from numpy.random import rand
from joblib import Parallel, delayed

class elastic_glm_regression:
    """
    This class provides elastic glm regression for functional data using the
    SRVF framework accounting for warping
    
    Usage:  obj = elastic_glm_regression(f,y,time)
    
    :param f: (M,N) % matrix defining N functions of M samples
    :param y: response vector of length N
    :param time: time vector of length M
    :param alpha: intercept
    :param b: coefficient vector
    :param B: basis matrix
    :param lambda: regularization parameter
    :param SSE: sum of squared errors

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  18-Mar-2018
    """

    def __init__(self, f, y, time):
        """
        Construct an instance of the elastic_glm_regression class
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

    def calc_model(self, link='linear', B=None, lam=0, df=20, max_itr=20, smooth_data=False, sparam=25, parallel=False):
        """
        This function identifies a regression model with phase-variability
        using elastic pca

        :param link: string of link function ('linear', 'quadratic', 'cubic')
        :param B: optional matrix describing Basis elements
        :param lam: regularization parameter (default 0)
        :param df: number of degrees of freedom B-spline (default 20)
        :param max_itr: maximum number of iterations (default 20)
        :param smooth_data: smooth data using box filter (default = F)
        :param sparam: number of times to apply box filter (default = 25)
        :param parallel: run in parallel (default = F)
        """
        if smooth_data:
            self.f = fs.smooth_data(self.f,sparam)
        
        print("Link: %s" % link)
        print("Lambda: %5.1f" % lam)

        self.lam = lam
        self.link = link

        # Create B-Spline Basis if none provided
        if B is None:
            B = bs(self.time, df=df, degree=4, include_intercept=True)
        Nb = B.shape[1]
        self.B = B

        n = self.f.shape[1]

        print("Initializing")
        b0 = rand(Nb+1)        
        out = minimize(MyLogLikelihoodFn, b0, args=(self.y,self.B,self.time,self.f,parallel), method="SLSQP")

        a = out.x

        if self.link == 'linear':
            h1, c_hat, cost = Amplitude_Index(self.f, self.time, self.B, self.y, max_itr, a, 1, parallel)
            yhat1 = c_hat[0] + MapC_to_y(n,c_hat[1:],self.B,self.time,self.f,parallel)
            yhat = np.polyval(h1,yhat1)
        elif self.link == 'quadratic':
            h1, c_hat, cost = Amplitude_Index(self.f, self.time, self.B, self.y, max_itr, a, 2, parallel)
            yhat1 = c_hat[0] + MapC_to_y(n,c_hat[1:],self.B,self.time,self.f,parallel)
            yhat = np.polyval(h1,yhat1)
        elif self.link == 'cubic':
            h1, c_hat, cost = Amplitude_Index(self.f, self.time, self.B, self.y, max_itr, a, 3, parallel)
            yhat1 = c_hat[0] + MapC_to_y(n,c_hat[1:],self.B,self.time,self.f,parallel)
            yhat = np.polyval(h1,yhat1)
        else:
            raise Exception('Invalid Link')
        
        tmp = (self.y-yhat)**2
        self.SSE = tmp.sum()
        self.h = h1
        self.alpha = c_hat[0]
        self.b = c_hat[1:]

        return

    def predict(self, newdata=None, parallel=True):
        """
        This function performs prediction on regression model on new data if available or current stored data in object
        Usage:  obj.predict()
                obj.predict(newdata)

        :param newdata: dict containing new data for prediction (needs the keys below, if None predicts on training data)
        :type newdata: dict
        :param f: (M,N) matrix of functions
        :param time: vector of time points
        :param y: truth if available
        :param smooth: smooth data if needed
        :param sparam: number of times to run filter
        """

        if newdata != None:
            f = newdata['f']
            time = newdata['time']
            y = newdata['y']
            sparam = newdata['sparam']
            if newdata['smooth']:
                f = fs.smooth_data(f,sparam)

            n = f.shape[1]
            yhat1 = self.alpha + MapC_to_y(n,self.b,self.B,time,f,parallel)
            yhat = np.polyval(self.h,yhat1)

            if y is None:
                self.SSE = np.nan
            else:
                self.SSE = np.sum((y-yhat)**2)
            
            self.y_pred = yhat

        else:
            n = self.f.shape[1]
            yhat1 = self.alpha + MapC_to_y(n,self.b,self.B,self.time,self.f,parallel)
            yhat = np.polyval(self.h,yhat1)
            self.SSE = np.sum((self.y-yhat)**2)
            self.y_pred = yhat

        return


def Amplitude_Index(f, t, B, y0, MaxIter, b, link, parallel):

    J = B.shape[1]
    n = f.shape[1]
    c_hat = b
    cost = 10000
    itr = 0
    while itr < MaxIter:
        itr += 1
        print("updating step: iter=%d" % (itr))
        y = c_hat[0] + MapC_to_y(n,c_hat[1:],B,t,f,parallel)
        h = np.polyfit(y, y0, link)
        b0 = rand(J+1)
        out = minimize(MyLogLikelihoodFn2, b0, args=(y0,B,t,f,h,parallel), method="SLSQP")

        if cost > out.fun:
            cost = out.fun
            c_hat = out.x
        else:
            c_hat = out.x
            break

    return(h,c_hat,cost)

def MyLogLikelihoodFn2(c, y0, B, t, f, h, parallel):

    N = f.shape[1]
    J = c.shape[0]
    y = c[0] + MapC_to_y(N,c[1:],B,t,f,parallel)
    tmp = np.polyval(h,y)
    x = (y0-tmp)*(y0-tmp)

    return(x.sum())

def MyLogLikelihoodFn(c, y0, B, t, f, parallel):

    N = f.shape[1]
    J = c.shape[0]
    y = c[0] + MapC_to_y(N,c[1:],B,t,f,parallel)
    x = (y0-y)*(y0-y)

    return(x.sum())
        
def MapC_to_y(n,c,B,t,f,parallel):

    dt = np.diff(t)
    dt = dt.mean()

    y = np.zeros(n)

    if parallel:
        bet = np.dot(B,c)
        q1 = uf.f_to_srsf(bet, t)
        y = Parallel(n_jobs=-1)(delayed(map_driver)(q1,
                                            f[:,k], bet, t, dt) for k in range(n))
    else:
        for k in range(0,n):
            bet = np.dot(B,c)
            q1 = uf.f_to_srsf(bet, t)
            q2 = uf.f_to_srsf(f[:,k], t)
            gam = uf.optimum_reparam(q1,t,q2)
            fn = uf.warp_f_gamma(t, f[:,k], gam)
            tmp = bet*fn
            y[k] = tmp.sum()*dt
    
    return(y)

def map_driver(q1, f, bet, t, dt):
    q2 = uf.f_to_srsf(f, t)
    gam = uf.optimum_reparam(q1,t,q2)
    fn = uf.warp_f_gamma(t, f, gam)
    tmp = bet*fn
    y = tmp.sum()*dt

    return y
