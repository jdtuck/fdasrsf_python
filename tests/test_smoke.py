"""Import / smoke tests: the package must import and expose its public API.

These are pure-python (no compiled extension, no sample data) so they run
everywhere and fail fast if a re-export in ``fdasrsf/__init__.py`` breaks.
"""

import importlib

import pytest


def test_package_imports_and_has_version():
    import fdasrsf as fs

    assert isinstance(fs.__version__, str)
    assert fs.__version__.count(".") >= 1


# Names re-exported at the top level of ``fdasrsf`` that must resolve.
PUBLIC_NAMES = [
    "fdawarp",
    "pairwise_align_functions",
    "optimum_reparam",
    "optimum_reparam_pair",
    "f_to_srsf",
    "srsf_to_f",
    "elastic_distance",
    "invertGamma",
    "smooth_data",
    "SqrtMean",
    "SqrtMeanInverse",
    "SqrtMedian",
    "cumtrapzmid",
    "rgam",
    "innerprod_q",
    "warp_f_gamma",
    "warp_q_gamma",
    "fdavpca",
    "fdahpca",
    "fdajpca",
    "pls_svd",
    "fdacurve",
    "curve_to_q",
    "q_to_curve",
    "calculatecentroid",
    "resamplecurve",
    "find_best_rotation",
    "elastic_distance_curve",
    "exp_map",
    "inv_exp_map",
    "apply_gam_imag",
    "makediffeoid",
    "rlbfgs",
]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_is_exported(name):
    import fdasrsf as fs

    assert hasattr(fs, name), f"fdasrsf.{name} is missing from the public API"


# Submodules advertised in ``fdasrsf.__all__`` must be importable.
SUBMODULES = [
    "time_warping",
    "utility_functions",
    "curve_stats",
    "curve_functions",
    "geometry",
    "geodesic",
    "fPCA",
    "fPLS",
    "gp",
    "boxplots",
    "kmeans",
    "tolerance",
    "image_functions",
    "regression",
    "pcr_regression",
    "elastic_changepoint",
]


@pytest.mark.parametrize("mod", SUBMODULES)
def test_submodule_imports(mod):
    importlib.import_module(f"fdasrsf.{mod}")
