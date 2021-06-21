import numpy as np
from numpy.random import multivariate_normal, rand, normal
from numpy.linalg import solve, det
from scipy.integrate import trapz, cumtrapz
from scipy.stats import truncnorm, norm
import fdasrsf.utility_functions as uf
import cbayesian as bay


def f_warp_pw(v, q1, q2):
    obs_domain = np.linspace(0,1,v.shape[0])
    exp1g_temp = uf.f_predictfunction(uf.f_exp1(v), obs_domain, 0)
    pt = np.insert(bay.bcuL2norm2(obs_domain, exp1g_temp),0,0)
    tmp = uf.f_predictfunction(q2,pt,0)
    out = tmp * exp1g_temp

    return(out)


def f_SSEv_pw(v, q1, q2):
    q2_gamma = f_warp_pw(v, q1, q2)
    vec = (q1 - q2_gamma)**2
    out = vec.sum()

    return(out)


def f_vpostlogl_pw(v, q1, q2, var, SSEv):
    if SSEv==0:
        SSEv = f_SSEv_pw(v, q1, q2)
    
    n = q1.shape[0]
    out = -n * np.log(np.sqrt(var)) - SSEv / (2 * var)
    return out, SSEv


def f_f1postlogl_pw(f1,y1,SSEv,K_f1,sigma_curr,sigma1_curr):
    n = y1.shape[0]
    iSig_f1 = K_f1 + np.eye(n) * n/sigma1_curr
    f1_mean = solve(iSig_f1, y1) * n/sigma1_curr
    SSEf1 = (f1-f1_mean) @ iSig_f1 @ (f1 - f1_mean)
    out = -SSEv/(2 * sigma_curr) - SSEf1/2
    return(out)


def f_f2postlogl_pw(f2,y2,SSEv,K_f2,sigma_curr,sigma2_curr):
    n = y2.shape[0]
    iSig_f2 = K_f2 + np.eye(n) * n/sigma2_curr
    f2_mean = solve(iSig_f2, y2) * n/sigma2_curr
    SSEf2 = (f2-f2_mean) @ iSig_f2 @ (f2 - f2_mean)
    out = -SSEv/(2 * sigma_curr) - SSEf2/2
    return(out)   


def f_dlogl_pw(v_coef, v_basis, d_basis, sigma_curr, q1, q2):
    vec = uf.f_basistofunction(v_basis["x"], 0, v_coef, v_basis)
    psi = uf.f_exp1(vec)
    N = q1.shape[0]
    obs_domain = np.linspace(0,1,N)
    binsize = np.diff(obs_domain)
    binsize = binsize.mean()
    gamma = uf.f_phiinv(psi)
    q2_warp = uf.warp_q_gamma(obs_domain, q2, gamma)
    q2_warp_grad = np.gradient(q2_warp, binsize)

    basismat = d_basis["matrix"]

    g = np.zeros(N)
    for i in range(0, basismat.shape[1]):
        ubar = cumtrapz(basismat[:,i], obs_domain, initial=0)
        integrand = (q1-q2_warp) * (-2 * q2_warp_grad*ubar-q2_warp*basismat[:,i])
        tmp = 1/sigma_curr * trapz(integrand, obs_domain)
        g += tmp * basismat[:,i]
    
    out, SSEv = f_vpostlogl_pw(vec, q1, q2, sigma_curr, 0)
    
    nll = -1 * out
    g_coef = v_basis["matrix"].T @ g

    return nll, g_coef, SSEv


def f_updatef1_pw(f1_curr, q1_curr, y1, q2, v_coef_curr, v_basis, SSE_curr, K_f1, K_f1prop, sigma_curr, sigma1_curr):
    time = np.linspace(0, 1, y1.shape[0])
    v = uf.f_basistofunction(v_basis["x"], 0, v_coef_curr, v_basis)

    f1_prop = multivariate_normal(f1_curr, K_f1prop)
    q1_prop = uf.f_to_srsf(f1_prop, time)

    SSE_prop = f_SSEv_pw(v, q1_prop, q2)

    postlog_curr = f_f1postlogl_pw(f1_curr, y1, SSE_curr, K_f1, sigma_curr, sigma1_curr)
    postlog_prop = f_f1postlogl_pw(f1_prop, y1, SSE_prop, K_f1, sigma_curr, sigma1_curr)

    ratio = np.minimum(1, np.exp(postlog_prop-postlog_curr))

    u = rand()
    if (u <= ratio):
        f1_curr = f1_prop
        q1_curr = q1_prop
        f1_accept = True
    else:
        f1_accept = False
    
    return f1_curr, q1_curr, f1_accept


def f_updatef2_pw(f2_curr, q2_curr, y2, q1, v_coef_curr, v_basis, SSE_curr, K_f2, K_f2prop, sigma_curr, sigma2_curr):
    time = np.linspace(0, 1, y2.shape[0])
    v = uf.f_basistofunction(v_basis["x"], 0, v_coef_curr, v_basis)

    f2_prop = multivariate_normal(f2_curr, K_f2prop)
    q2_prop = uf.f_to_srsf(f2_prop, time)

    SSE_prop = f_SSEv_pw(v, q1, q2_prop)

    postlog_curr = f_f2postlogl_pw(f2_curr, y2, SSE_curr, K_f2, sigma_curr, sigma2_curr)
    postlog_prop = f_f2postlogl_pw(f2_prop, y2, SSE_prop, K_f2, sigma_curr, sigma2_curr)

    ratio = np.minimum(1, np.exp(postlog_prop-postlog_curr))

    u = rand()
    if (u <= ratio):
        f2_curr = f2_prop
        q2_curr = q2_prop
        f2_accept = True
    else:
        f2_accept = False
    
    return f2_curr, q2_curr, f2_accept


def f_updatephi_pw(f1_curr, K_f1, s1_curr, L1_curr, L1_propvar, Dmat):
    a, b = (L1_curr - L1_curr) / L1_propvar, (100000 - L1_curr) / L1_propvar
    L1_prop = truncnorm.rvs(a, b)

    K_f1_tmp = s1_curr * (uf.exp2corr2(L1_prop,Dmat) +0.1 * np.eye(f1_curr.shape[0]))

    SSEf_curr = (f1_curr @ K_f1 @ f1_curr)/2
    SSEf_prop = (uf.mrdivide(f1_curr,K_f1_tmp) @ f1_curr.T)/2

    postlog_prop = -np.log(det(K_f1_tmp))/2 - SSEf_prop - np.log(1-norm.cdf(-L1_curr/L1_propvar))
    postlog_curr = np.log(det(K_f1))/2 - SSEf_curr - np.log(1-norm.cdf(-L1_prop/L1_propvar))
    ratio = np.minimum(1, np.exp(postlog_prop-postlog_curr))

    u = rand()
    if (u <= ratio):
        L1_curr = L1_prop
        L1_accept = True
    else:
        L1_accept = False
    
    return L1_curr, L1_accept


def f_updatev_pw(v_coef_curr, v_basis, sigma_curr, q1, q2, nll_cur, g_cur, SSE_curr, propose_v_coef, d_basis, cholC, h, L):
    v_coef_prop = propose_v_coef(v_coef_curr)

    q = v_coef_prop.copy()
    D = q.shape[0]
    rth = np.sqrt(h)
    g_cur = -1*g_cur

    # sample velocity
    v = normal(size=D)
    v = cholC.T@v

    # natural gradient
    halfng = cholC@g_cur
    ng = cholC@halfng

    # accumulate the power of force
    pow = rth/2*(np.dot(g_cur,v))

    # calculate current energy
    E_cur = nll_cur - h/8*(np.dot(halfng,halfng))

    randL = int(np.ceil(rand()*L))
    
    # Alternate full sth for position and velocity
    for l in range(0, randL):
        # Make a half step for velocity
        v = v + rth/2 * ng

        # Make a full step for position
        rot = (q+1j*v)*np.exp(-1j*rth)
        q = np.real(rot)
        v = np.imag(rot)

        # update geometry
        nll, g, SSEv = f_dlogl_pw(q, v_basis, d_basis, sigma_curr, q1, q2)
        halfng = cholC@g
        ng = cholC.T@halfng

        # Make a half step for velocity
        v = v + rth/2*ng

        # accumulate the power of force
        if l!=randL:
            pow = pow + rth*(np.dot(g,v))
    
    # accumulate the power of force
    pow = pow + rth/2 * (np.dot(g,v))

    # evaluate energy at start and end of trajectory
    E_prop = nll - h/8*(np.dot(halfng,halfng))

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    logRatio = -E_prop + E_cur - pow

    if (not np.isinf(logRatio)) and (np.log(rand()) < np.minimum(0, logRatio)):
        v_coef_curr = q
        theta_accept = True
        SSE_curr = SSEv
    else:
        nll = nll_cur
        g = g_cur
        theta_accept = False


    return v_coef_curr, nll, g, SSE_curr, theta_accept


def f_SSEg_pw(g, q1, q2):
    obs_domain = np.linspace(0,1,g.shape[0])
    exp1g_temp = uf.f_predictfunction(uf.f_exp1(g), obs_domain, 0)
    pt = np.insert(bay.bcuL2norm2(obs_domain, exp1g_temp),0,0)
    tmp = uf.f_predictfunction(q2,pt,0)
    vec = (q1 - tmp * exp1g_temp)**2
    out = vec.sum()
    return(out)


def f_logl_pw(g, q1, q2, var1, SSEg):
    if SSEg == 0:
        SSEg = f_SSEg_pw(g, q1, q2)

    n = q1.shape[0]
    out = n * np.log(1/np.sqrt(2*np.pi)) - n * np.log(np.sqrt(var1)) - SSEg / (2 * var1)

    return(out)


def f_updateg_pw(g_coef_curr,g_basis,var1_curr,q1,q2,SSE_curr,propose_g_coef):
    g_coef_prop = propose_g_coef(g_coef_curr)

    tst = uf.f_exp1(uf.f_basistofunction(g_basis["x"],0,g_coef_prop["prop"],g_basis))

    while tst.min() < 0:
        g_coef_prop = propose_g_coef(g_coef_curr)
        tst = uf.f_exp1(uf.f_basistofunction(g_basis["x"],0,g_coef_prop["prop"],g_basis))

    if SSE_curr == 0:
        SSE_curr = f_SSEg_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_curr,g_basis), q1, q2)

    SSE_prop = f_SSEg_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_prop["prop"],g_basis), q1, q2)

    logl_curr = f_logl_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_curr,g_basis), q1, q2, var1_curr, SSE_curr)
    
    logl_prop = f_logl_pw(uf.f_basistofunction(g_basis["x"],0,g_coef_prop["prop"],g_basis), q1, q2, var1_curr, SSE_prop)

    ratio = np.minimum(1, np.exp(logl_prop-logl_curr))

    u = np.random.rand()
    if u <= ratio:
        g_coef = g_coef_prop["prop"]
        logl = logl_prop
        SSE = SSE_prop
        accept = True
        zpcnInd = g_coef_prop["ind"]

    if u > ratio:
        g_coef = g_coef_curr
        logl = logl_curr
        SSE = SSE_curr
        accept = False
        zpcnInd = g_coef_prop["ind"]

    return g_coef, logl, SSE, accept, zpcnInd
