import numpy as np
from scipy.integrate import trapz, cumtrapz
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
    f1_mean = np.linalg.solve(iSig_f1, y1) * n/sigma1_curr
    SSEf1 = (f1-f1_mean) @ iSig_f1 @ (f1 - f1_mean)
    out = -SSEv/(2 * sigma_curr) - SSEf1/2
    return(out)


def f_f2postlogl_pw(f2,y2,SSEv,K_f2,sigma_curr,sigma2_curr):
    n = y2.shape[0]
    iSig_f2 = K_f2 + np.eye(n) * n/sigma2_curr
    f2_mean = np.linalg.solve(iSig_f2, y2) * n/sigma2_curr
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
    g_coef = v_basis["matrix"] @ g

    return nll, g_coef, SSEv


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
