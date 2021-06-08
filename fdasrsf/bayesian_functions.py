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