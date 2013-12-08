"""
A python package for functional data analysis using the square root
slope framework which performs pair-wise and group-wise
alignment as well as modeling using functional component
analysis

"""
__all__ = ["time_warping", "utility_functions", "fPCA"]

__version__ = "1.1.0"

import sys

if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError("Python Version 2.6 or above is required for fdasrsf.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .time_warping import srsf_align, srsf_align_pair, align_fPCA, align_fPLS
from .plot_style import f_plot, rstyle
from .utility_functions import smooth_data, optimum_reparam, f_to_srsf, gradient_spline, elastic_distance, invertGamma
from .utility_functions import SqrtMean, SqrtMeanInverse, cumtrapzmid, rgam, outlier_detection, innerprod_q
from .utility_functions import optimum_reparam_pair, f_K_fold, zero_crossing, warp_q_gamma
from .fPCA import vertfPCA, horizfPCA
from .gauss_model import gauss_model
from .fPLS import pls_svd
from .regression import elastic_prediction, elastic_logistic, elastic_regression
