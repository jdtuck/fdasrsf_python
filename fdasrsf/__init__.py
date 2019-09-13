"""
A python package for functional data analysis using the square root
slope framework which performs pair-wise and group-wise
alignment as well as modeling using functional component
analysis

"""
__all__ = ["time_warping", "utility_functions", "curve_functions", "geometry", "pcr_regression", "tolerance"]

__version__ = "1.5.0"

import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise ImportError("Python Version 3.6 or above is required for fdasrsf.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .time_warping import fdawarp, align_fPCA, align_fPLS
from .plot_style import f_plot, rstyle, plot_reg_open_curve, plot_geod_open_curve, plot_geod_close_curve
from .utility_functions import smooth_data, optimum_reparam, f_to_srsf, gradient_spline, elastic_distance, invertGamma, srsf_to_f
from .utility_functions import SqrtMean, SqrtMeanInverse, cumtrapzmid, rgam, outlier_detection, innerprod_q
from .utility_functions import optimum_reparam_pair, f_K_fold, zero_crossing, warp_q_gamma, resamplefunction, warp_f_gamma
from .fPCA import fdavpca, fdahpca, fdajpca
from .gauss_model import gauss_model, joint_gauss_model
from .fPLS import pls_svd
from .regression import elastic_prediction, elastic_logistic, elastic_regression, elastic_mlogistic
from .pcr_regression import elastic_pcr_regression, elastic_lpcr_regression, elastic_mlpcr_regression
from .boxplots import ampbox, phbox
from .tolerance import bootTB, pcaTB
from .curve_functions import resamplecurve, calculatecentroid, innerprod_q2, curve_to_q, optimum_reparam_curve, find_best_rotation
from .curve_functions import find_rotation_and_seed_coord, group_action_by_gamma_coord, group_action_by_gamma, q_to_curve, rot_mat
from .curve_functions import calculate_variance, project_curve, pre_proc_curve, inverse_exp_coord, project_tangent, gram_schmidt, scale_curve, parallel_translate, curve_zero_crossing
from .geodesic import geod_sphere, path_straightening
from .curve_stats import curve_karcher_mean, curve_karcher_cov, curve_principal_directions, sample_shapes, oc_srvf_align
from .curve_regression import oc_elastic_logistic, oc_elastic_prediction, preproc_open_curve, oc_elastic_mlogistic
from .geometry import inv_exp_map, exp_map
