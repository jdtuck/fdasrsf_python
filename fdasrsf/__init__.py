"""
A python package for functional data analysis using the square root
slope framework which performs pair-wise and group-wise
alignment as well as modeling using functional component
analysis

"""
__all__ = ["time_warping", "utility_functions", "bayesian_functions", "curve_stats", "geodesic", "curve_functions", 
           "geometry", "pcr_regression", "tolerance", "boxplots", "curve_regression", "regression", "fPCA", 
           "elastic_glm_regression", "curve_pcr_regression", "kmeans", "image", "image_functions"]

__version__ = "2.4.0"

import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise ImportError("Python Version 3.6 or above is required for fdasrsf.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .time_warping import fdawarp, align_fPCA, align_fPLS, pairwise_align_bayes, pairwise_align_functions
from .time_warping import pairwise_align_bayes_infHMC
from .plot_style import f_plot, rstyle, plot_curve, plot_reg_open_curve, plot_geod_open_curve, plot_geod_close_curve
from .utility_functions import smooth_data, optimum_reparam, f_to_srsf, gradient_spline, elastic_distance, invertGamma, srsf_to_f
from .utility_functions import SqrtMean, SqrtMeanInverse, cumtrapzmid, rgam, outlier_detection, innerprod_q
from .utility_functions import optimum_reparam_pair, warp_q_gamma, resamplefunction, warp_f_gamma, elastic_depth
from .fPCA import fdavpca, fdahpca, fdajpca
from .fPLS import pls_svd
from .regression import elastic_logistic, elastic_regression, elastic_mlogistic
from .elastic_glm_regression import elastic_glm_regression
from .pcr_regression import elastic_pcr_regression, elastic_lpcr_regression, elastic_mlpcr_regression
from .curve_pcr_regression import elastic_curve_pcr_regression
from .boxplots import ampbox, phbox
from .tolerance import bootTB, pcaTB
from .kmeans import kmeans_align
from .image import reparam_image
from .image_functions import formbasisTid, apply_gam_imag, makediffeoid
from .curve_functions import resamplecurve, calculatecentroid, curve_to_q, optimum_reparam_curve, find_best_rotation, elastic_distance_curve
from .curve_functions import q_to_curve, find_rotation_and_seed_coord
from .curve_stats import fdacurve
from .curve_regression import oc_elastic_regression, oc_elastic_logistic, preproc_open_curve, oc_elastic_mlogistic
from .geometry import inv_exp_map, exp_map
from .geodesic import geod_sphere, path_straightening
from .umap_metric import efda_distance, efda_distance_curve
from .rbfgs import rlbfgs
