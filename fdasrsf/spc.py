"""
Elastic Functional Statistical Process Control

PCA-based process monitoring (Hotelling T^2 and SPE/Q charts) of functional
data in the SRSF framework, with separate models for amplitude (vertical
fPCA) and phase (horizontal fPCA). The monitoring statistics and control
limits follow Z. Ge and Z. Song, "Multivariate Statistical Process
Control", Springer 2013, Sec. 2.1, Eqs. (2.3)-(2.8), applied to the elastic
fPCA features.

Reference:
    J. D. Tucker, "Functional Statistical Process Control using Elastic
    Methods," Proc. JSM, August 2016.
    https://research.tetonedge.net/Papers/JSM16.pdf

moduleauthor:: J. Derek Tucker <jdtuck@sandia.gov>

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist, chi2
import fdasrsf.geometry as geo
from fdasrsf.fPCA import fdavpca, fdahpca


class fdaspc:
    """
    This class provides elastic functional statistical process control

    Usage:  obj = fdaspc(warp_data)
            obj.train()
            out = obj.monitor(f_new)

    :param warp_data: fdawarp class with alignment data of the
                      normal-operating-condition functions
    :param vpca: fdavpca object (amplitude model)
    :param hpca: fdahpca object (phase model)
    :param alpha: significance level of the control limits
    :param t2limit: T^2 limit formula, 'standard' or 'book'
    :param amp: amplitude model: k, n, T2lim, SPElim, g, h, training T2/SPE
    :param ph: phase model: k, n, T2lim, SPElim, g, h, training T2/SPE
    :param mon: results of the last call to monitor

    Author :  J. D. Tucker (JDT) <jdtuck AT sandia.gov>
    Date   :  20-Nov-2026
    """

    def __init__(self, fdawarp):
        """
        Construct an instance of the fdaspc class

        :param fdawarp: fdawarp class aligned with srsf_align
        """
        if getattr(fdawarp, "fn", None) is None:
            raise Exception("Please align fdawarp class using srsf_align!")

        self.warp_data = fdawarp
        self.amp = None
        self.ph = None
        self.mon = None

    def train(
        self,
        var_exp=0.95,
        no=None,
        alpha=0.01,
        t2limit="standard",
        id=None,
        log_der=False,
        realign=True,
        smoothdata=None,
    ):
        """
        Fit the amplitude and phase monitoring models

        Runs vertical and horizontal fPCA on the aligned data and computes
        the Hotelling T^2 limit (Ge & Song Eq. 2.7) and SPE limit (Eq. 2.8,
        Box chi-square approximation) for each.

        Usage: obj.train()
               obj.train(var_exp=0.9, alpha=0.05)
               obj.train(no=[3, 2])

        :param var_exp: cumulative variance used to choose the number of
                        components (default = 0.95); scalar, or [amp, phase]
        :param no: number of components; scalar or [amp, phase]. When
                   given, it overrides var_exp (default = None)
        :param alpha: significance level of the limits (default = 0.01)
        :param t2limit: 'standard', k(n^2-1)/(n(n-k)) F, or 'book',
                        k(n-1)/(n-k) F (default = 'standard')
        :param id: point used for f(id) in the amplitude model
                   (default = midpoint)
        :param log_der: use the log-derivative transform for the phase model
                        instead of the shooting vectors (default = False)
        :param realign: compute the training T^2/SPE and the limits from the
                        training functions re-aligned to the template with
                        align_new, the same path monitor uses (default =
                        True). Re-alignment perturbs the srsfs, so limits fit
                        to the in-sample residuals are too tight for
                        monitored data; False uses the in-sample features
        :param smoothdata: smooth the data when computing the srsfs for the
                           re-alignment (default = None, the value used for
                           the alignment fit, see fdawarp.align_new)
        """
        if not (0 < alpha < 1):
            raise Exception("alpha must be in (0, 1)")
        if t2limit not in ("standard", "book"):
            raise Exception("t2limit must be 'standard' or 'book'")
        var_exp = _pair(var_exp)
        no = _pair(no)
        for i in range(2):
            if no[i] is not None:
                var_exp[i] = None

        self.alpha = alpha
        self.t2limit = t2limit

        # amplitude: vertical fPCA on [qn; sign(f(id))sqrt|f(id)|]
        self.vpca = fdavpca(self.warp_data)
        self.vpca.calc_fpca(
            no=3 if no[0] is None else int(no[0]), var_exp=var_exp[0], id=id
        )

        # phase: horizontal fPCA on the shooting vectors
        self.hpca = fdahpca(self.warp_data, log_der)
        self.hpca.calc_fpca(no=3 if no[1] is None else int(no[1]), var_exp=var_exp[1])

        # features the limits are fit to
        if realign:
            fn, qn, gam = self.warp_data.align_new(
                self.warp_data.f, smoothdata=smoothdata
            )
            X = _amp_features(self.vpca, fn, qn)
            V = _ph_features(self.hpca, gam)
        else:
            X = _amp_features(self.vpca, self.warp_data.fn, self.warp_data.qn)
            V = self.hpca.vec

        self.amp = _fit_limits(
            X, self.vpca.mqn, self.vpca.U, self.vpca.latent, alpha, t2limit
        )
        self.ph = _fit_limits(
            V, self.hpca.vm, self.hpca.U, self.hpca.latent, alpha, t2limit
        )
        self.mon = None

        return

    def monitor(self, f_new, smoothdata=None):
        """
        Monitor new functions against the trained models

        Aligns the new functions to the training template (same lambda,
        method and penalty) and computes the scores, T^2, SPE, alarms and
        per-variable contributions for amplitude and phase. A fault is
        detected when T^2 > T2lim or SPE > SPElim.

        Usage: out = obj.monitor(f_new)

        :param f_new: numpy ndarray of shape (M,N) of N new functions sampled
                      at warp_data.time
        :param smoothdata: smooth the data when computing the srsfs
                           (default = None, see fdawarp.align_new)

        :rtype: dict (also stored in obj.mon)
        :return fn, qn, gam: aligned functions, srsfs and warps
        :return amp, ph: dicts with T (scores), T2, SPE, alarmT2, alarmSPE,
                         alarm, contT2, contSPE (rows are new functions;
                         contributions are by time point, the amplitude ones
                         have the f(id) term last), T2lim and SPElim
        :return alarm: (N,) bool, any amplitude or phase alarm
        """
        if self.amp is None:
            raise Exception("Please train the fdaspc object first!")

        fn, qn, gam = self.warp_data.align_new(f_new, smoothdata=smoothdata)

        # amplitude
        X = _amp_features(self.vpca, fn, qn)
        amp_out = _monitor_stats(
            X, self.vpca.mqn, self.vpca.U, self.vpca.latent, self.amp
        )

        # phase
        V = _ph_features(self.hpca, gam)
        ph_out = _monitor_stats(
            V, self.hpca.vm, self.hpca.U, self.hpca.latent, self.ph
        )

        out = {
            "fn": fn,
            "qn": qn,
            "gam": gam,
            "amp": amp_out,
            "ph": ph_out,
            "alarm": amp_out["alarm"] | ph_out["alarm"],
        }
        self.mon = out

        return out

    def plot(self):
        """
        control charts for amplitude and phase

        T^2 and SPE charts of the training data, followed by the data from
        the last call to monitor; dashed lines are the control limits and
        red points are alarms.

        Usage: obj.plot()
        """
        if self.amp is None:
            raise Exception("Please train the fdaspc object first!")
        mods = [self.amp, self.ph]
        names = ["Amplitude", "Phase"]
        stats = ["T2", "SPE"]
        labels = ["$T^2$", "SPE"]
        if self.mon is None:
            news = [None, None]
        else:
            news = [self.mon["amp"], self.mon["ph"]]

        fig, axs = plt.subplots(2, 2, layout="constrained")
        for a in range(2):
            for s in range(2):
                ax = axs[a, s]
                tr = mods[a][stats[s]]
                lim = mods[a][stats[s] + "lim"]
                ntr = tr.shape[0]
                ax.plot(np.arange(1, ntr + 1), tr, ".", color="0.5")
                if news[a] is not None:
                    nw = news[a][stats[s]]
                    idx = ntr + np.arange(1, nw.shape[0] + 1)
                    al = nw > lim
                    ax.plot(idx, nw, "b.-")
                    ax.plot(idx[al], nw[al], "o", color="r", markerfacecolor="r")
                    ax.axvline(ntr + 0.5, color="k", linestyle=":")
                ax.axhline(lim, color="k", linestyle="--", linewidth=1.5)
                ax.set_xlabel("Observation")
                ax.set_ylabel(labels[s])
                ax.set_title("%s %s (k = %d)" % (names[a], labels[s], mods[a]["k"]))

        return fig, axs

    def plot_contrib(self, idx):
        """
        contributions over time for a monitored function

        Plots the T^2 and SPE contributions of each time point for the
        amplitude and phase models, to locate where in time a fault occurs.

        Usage: obj.plot_contrib(idx)

        :param idx: index of the function passed to the last monitor call
        """
        if self.mon is None:
            raise Exception("Please call monitor first!")
        t = self.warp_data.time
        amp = self.mon["amp"]
        ph = self.mon["ph"]
        fig, axs = plt.subplots(2, 2, layout="constrained")
        axs[0, 0].plot(t, amp["contT2"][idx, :-1], linewidth=1.5)
        axs[0, 0].set_title("Amplitude $T^2$ contribution")
        axs[0, 1].plot(t, amp["contSPE"][idx, :-1], linewidth=1.5)
        axs[0, 1].set_title("Amplitude SPE contribution")
        axs[1, 0].plot(t, ph["contT2"][idx, :], linewidth=1.5)
        axs[1, 0].set_title("Phase $T^2$ contribution")
        axs[1, 1].plot(t, ph["contSPE"][idx, :], linewidth=1.5)
        axs[1, 1].set_title("Phase SPE contribution")
        for ax in axs.ravel():
            ax.set_xlabel("t")

        return fig, axs


def _pair(x):
    """scalar -> [x, x], keep a 2-element sequence"""
    if x is None or np.isscalar(x):
        return [x, x]
    x = list(x)
    if len(x) != 2:
        raise Exception("expected a scalar or a 2-element [amp, phase] vector")
    return x


def _amp_features(vpca, fn, qn):
    """columns [qn; sign(f(id))sqrt|f(id)|] as in fdavpca"""
    fid = fn[vpca.id, :]
    m = np.sign(fid) * np.sqrt(np.abs(fid))
    return np.vstack((qn, m))


def _ph_features(hpca, gam):
    """
    shooting vectors at the Karcher mean (same map as uf.SqrtMean, which
    fdahpca.calc_fpca uses), or the log-derivative transform
    """
    if hpca.log_der:
        return geo.gam_to_h(gam)
    M, n = gam.shape
    binsize = 1.0 / (M - 1)
    V = np.zeros((M, n))
    for i in range(n):
        psi = np.sqrt(np.maximum(np.gradient(gam[:, i], binsize), 0))
        V[:, i] = geo.inv_exp_map(hpca.mu_psi, psi)[0]
    return V


def _scores(X, mu, U, latent):
    """Ge & Song Eqs. (2.3)-(2.6) and contributions; X is (p,N)"""
    Xc = (X - np.asarray(mu)[:, np.newaxis]).T  # N x p
    T = Xc @ U  # scores
    E = Xc - T @ U.T  # residuals
    lam = np.asarray(latent)
    T2 = (T**2 / lam).sum(axis=1)
    SPE = (E**2).sum(axis=1)
    D = U / np.sqrt(lam)
    contT2 = ((Xc @ D) @ D.T) * Xc  # rows sum to T2
    contSPE = E**2  # rows sum to SPE
    return {"T": T, "T2": T2, "SPE": SPE, "contT2": contT2, "contSPE": contSPE}


def _fit_limits(X, mu, U, latent, alpha, t2limit):
    """control limits from the training data, Ge & Song Eqs. (2.7)-(2.8)"""
    n = X.shape[1]
    k = U.shape[1]
    if k >= n:
        raise Exception(
            "Need fewer components than functions for the T^2 limit "
            "(k = %d, n = %d)." % (k, n)
        )
    S = _scores(X, mu, U, latent)

    Fcrit = f_dist.ppf(1 - alpha, k, n - k)
    if t2limit == "book":
        T2lim = k * (n - 1) / (n - k) * Fcrit
    else:
        T2lim = k * (n**2 - 1) / (n * (n - k)) * Fcrit

    mSPE = S["SPE"].mean()
    vSPE = S["SPE"].var(ddof=1)
    eps = np.finfo(np.double).eps
    if vSPE < eps:
        g = eps
        h = 1.0
    else:
        g = vSPE / (2 * mSPE)
        h = 2 * mSPE**2 / vSPE
    SPElim = g * chi2.ppf(1 - alpha, h)

    return {
        "k": k,
        "n": n,
        "T2lim": T2lim,
        "SPElim": SPElim,
        "g": g,
        "h": h,
        "T2": S["T2"],
        "SPE": S["SPE"],
    }


def _monitor_stats(X, mu, U, latent, mdl):
    out = _scores(X, mu, U, latent)
    out["alarmT2"] = out["T2"] > mdl["T2lim"]
    out["alarmSPE"] = out["SPE"] > mdl["SPElim"]
    out["alarm"] = out["alarmT2"] | out["alarmSPE"]
    out["T2lim"] = mdl["T2lim"]
    out["SPElim"] = mdl["SPElim"]
    return out
