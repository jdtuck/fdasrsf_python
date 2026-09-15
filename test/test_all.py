import unittest
import numpy as np
import fdasrsf as fs


class TestFDASRSF(unittest.TestCase):
    # Returns True or False.
    def test_reparm(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1)
        self.assertAlmostEqual(sum(gam - timet), 0)

    def test_warp_f_gamma(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1)
        q1a = fs.warp_f_gamma(timet, q1, gam)
        self.assertAlmostEqual(sum(q1 - q1a), 0)

    def test_warp_q_gamma(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1)
        q1a = fs.warp_q_gamma(timet, q1, gam)
        self.assertAlmostEqual(sum(q1 - q1a), 0)

    def test_reparm1(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1, method="DP")
        self.assertAlmostEqual(sum(gam - timet), 0)

    def test_rlbgs(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1, method="RBFGS")
        self.assertAlmostEqual(sum(gam - timet), 0)

    def test_crlbgs(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        gam = fs.optimum_reparam(q1, timet, q1, method="cRBFGS")
        self.assertAlmostEqual(sum(gam - timet), 0)

    # the DP solvers accumulate the penalty edge by edge, so they can only
    # carry the penalties that are an integral of a pointwise function of
    # gammadot
    DP_PENALTIES = ("none", "l2gam", "l2psi")
    NON_DP_PENALTIES = ("roughness", "geodesic")

    def test_reparm_penalties(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        for method in ("DP", "DP2", "RBFGS", "cRBFGS"):
            penalties = self.DP_PENALTIES
            if method not in ("DP", "DP2"):
                penalties = penalties + self.NON_DP_PENALTIES
            for penalty in penalties:
                with self.subTest(method=method, penalty=penalty):
                    gam = fs.optimum_reparam(
                        q1, timet, q1, method=method, lam=0.1, penalty=penalty
                    )
                    self.assertAlmostEqual(sum(gam - timet), 0)

    def test_reparm_penalty_without_dp_counterpart(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        for method in ("DP", "DP2"):
            for penalty in self.NON_DP_PENALTIES:
                with self.subTest(method=method, penalty=penalty):
                    # a nonzero weight is a request the DP solver cannot honour
                    with self.assertRaises(ValueError):
                        fs.optimum_reparam(
                            q1, timet, q1, method=method, lam=0.1, penalty=penalty
                        )
                    # ... but with lam == 0 the penalty drops out of the cost,
                    # so every method can satisfy the request exactly
                    gam = fs.optimum_reparam(
                        q1, timet, q1, method=method, lam=0.0, penalty=penalty
                    )
                    self.assertAlmostEqual(sum(gam - timet), 0)

    def test_reparm_batched_penalties(self):
        M = 101
        timet = np.linspace(0, 1, M)
        q1 = fs.f_to_srsf(np.sin(2 * np.pi * timet), timet)
        q2 = fs.f_to_srsf(np.sin(2 * np.pi * timet**2), timet)
        Q1 = np.column_stack((q1, q1))
        Q2 = np.column_stack((q2, q2))
        for method in ("DP", "DP2"):
            with self.subTest(method=method):
                gam = fs.optimum_reparam(
                    q1, timet, q2, method=method, lam=1.0, penalty="l2gam"
                )
                gamN = fs.optimum_reparam(
                    q1, timet, Q2, method=method, lam=1.0, penalty="l2gam"
                )
                gamN2 = fs.optimum_reparam(
                    Q1, timet, Q2, method=method, lam=1.0, penalty="l2gam"
                )
                np.testing.assert_allclose(gamN[:, 0], gam)
                np.testing.assert_allclose(gamN2[:, 1], gam)

    def test_umap_efda_distance(self):
        from fdasrsf.umap_metric import efda_distance

        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        q2 = np.cos(np.linspace(0, 2 * np.pi, M))
        d = efda_distance(q1, q2)
        self.assertTrue(np.isfinite(d))
        self.assertGreater(d, 0)

    def test_optimum_reparam_pair(self):
        # the pair is aligned jointly, so one warping comes back for both
        M = 101
        timet = np.linspace(0, 1, M)
        qa = fs.f_to_srsf(np.sin(2 * np.pi * timet), timet)
        qb = fs.f_to_srsf(np.cos(2 * np.pi * timet), timet)
        q = np.column_stack((qa, qb))
        q1 = fs.f_to_srsf(np.sin(2 * np.pi * timet**1.3), timet)
        q2 = fs.f_to_srsf(np.cos(2 * np.pi * timet**1.3), timet)

        # aligning a pair to itself gives the identity
        gamid = fs.optimum_reparam_pair(q, timet, qa, qb)
        np.testing.assert_allclose(gamid, timet, atol=1e-10)

        gam = fs.optimum_reparam_pair(q, timet, q1, q2)
        self.assertEqual(gam.shape, (M,))
        self.assertTrue(np.all(np.diff(gam) >= -1e-12))

        # the batched branch must agree column by column with the single one
        Q1 = np.column_stack((q1, q1))
        Q2 = np.column_stack((q2, q2))
        gamN = fs.optimum_reparam_pair(q, timet, Q1, Q2)
        self.assertEqual(gamN.shape, (M, 2))
        np.testing.assert_allclose(gamN[:, 0], gam)
        np.testing.assert_allclose(gamN[:, 1], gam)

        with self.assertRaises(ValueError):
            fs.optimum_reparam_pair(qa, timet, q1, q2)
        with self.assertRaises(ValueError):
            fs.optimum_reparam_pair(q, timet, q1, Q2)

    def test_apply_gam_imag(self):
        # the identity diffeomorphism must leave the image alone, and a
        # genuine 2-D one must sample it at the points it names: both fail if
        # the two components of gam are not flattened the same way
        m, n = 41, 33
        U = np.linspace(0, 1, m)
        V = np.linspace(0, 1, n)
        F = np.sin(2 * np.pi * U[:, None]) * np.cos(3 * np.pi * V[None, :])

        gamid = fs.makediffeoid(m, n)
        np.testing.assert_allclose(fs.apply_gam_imag(F, gamid), F, atol=1e-12)

        UU, VV = np.meshgrid(U, V, indexing="ij")
        bump = 0.1 * np.sin(np.pi * UU) * np.sin(np.pi * VV)
        gam = np.zeros((m, n, 2))
        gam[:, :, 0] = VV - bump
        gam[:, :, 1] = UU + bump
        expected = np.sin(2 * np.pi * gam[:, :, 1]) * np.cos(
            3 * np.pi * gam[:, :, 0]
        )
        # the residual is the bilinear interpolation error on this grid
        np.testing.assert_allclose(
            fs.apply_gam_imag(F, gam), expected, atol=2e-2
        )

    def test_reparm_bad_penalty(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        with self.assertRaises(ValueError):
            fs.optimum_reparam(q1, timet, q1, penalty="bogus")

    def test_f_to_srvf(self):
        M = 101
        f1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        q1 = fs.f_to_srsf(f1, timet)
        f1a = fs.srsf_to_f(q1, timet)
        self.assertAlmostEqual(sum(f1 - f1a), 0, 4)

    def test_elastic_distance(self):
        M = 101
        f1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        da, dp = fs.elastic_distance(f1, f1, timet)
        self.assertLessEqual(da, 1e-10)
        self.assertLessEqual(dp, 1e-6)

    def test_smooth(self):
        M = 101
        q1 = np.zeros((M, 1))
        q1[:, 0] = np.sin(np.linspace(0, 2 * np.pi, M)).T
        q1a = fs.smooth_data(q1, 1)
        q1b = fs.smooth_data(q1, 1)
        self.assertAlmostEqual(sum(q1a.flatten() - q1b.flatten()), 0)

    def test_edistance(self):
        M = 101
        q1 = np.sin(np.linspace(0, 2 * np.pi, M))
        timet = np.linspace(0, 1, M)
        dy, dx = fs.elastic_distance(q1, q1, timet)
        self.assertAlmostEqual(dy + dx, 0)

    def test_invgamma(self):
        M = 101
        gam = np.linspace(0, 1, M)
        gami = fs.invertGamma(gam)
        self.assertAlmostEqual(sum(gam - gami), 0)

    def test_invexpmap(self):
        M = 101
        gam = np.linspace(0, 1, M)
        binsize = np.mean(np.diff(gam))
        psi = np.sqrt(np.gradient(gam, binsize))
        out, theta = fs.inv_exp_map(psi, psi)
        self.assertAlmostEqual(sum(out), 0)

    def test_l2norm(self):
        M = 101
        gam = np.linspace(0, 1, M)
        binsize = np.mean(np.diff(gam))
        psi = np.sqrt(np.gradient(gam, binsize))
        out, theta = fs.inv_exp_map(psi, psi)
        out1 = fs.geometry.L2norm(out)
        self.assertAlmostEqual(out1, 0)

    def test_expmap(self):
        M = 101
        gam = np.linspace(0, 1, M)
        binsize = np.mean(np.diff(gam))
        psi = np.sqrt(np.gradient(gam, binsize))
        out, theta = fs.inv_exp_map(psi, psi)
        out1 = fs.exp_map(psi, out)
        self.assertAlmostEqual(sum(out1), M)

    def test_srsf(self):
        data = np.load("bin/simu_data.npz")
        time = data["arr_1"]
        f = data["arr_0"]
        obj = fs.fdawarp(f, time)
        obj.srsf_align()
        vpca = fs.fdavpca(obj)
        vpca.calc_fpca()
        vpca = fs.fdajpca(obj)
        vpca.calc_fpca()
        vpca = fs.fdahpca(obj)
        vpca.calc_fpca()
        self.assertAlmostEqual(obj.amp_var, 0.018998691036349585)

    def test_srvf(self):
        data = np.load("bin/MPEG7.npz", allow_pickle=True)
        Xdata = data["Xdata"]
        curve = Xdata[0, 1]
        n, M = curve.shape
        K = Xdata.shape[1]

        beta = np.zeros((n, M, K))
        for i in range(0, K):
            beta[:, :, i] = Xdata[0, i]

        obj = fs.fdacurve(beta, N=M)
        obj.karcher_mean()
        obj.srvf_align()
        obj.karcher_cov()
        obj.shape_pca()
        self.assertAlmostEqual(obj.E[-1], 0.022668183569717587)


if __name__ == "__main__":
    unittest.main()
