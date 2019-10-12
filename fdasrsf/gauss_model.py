"""
Gaussian Model of functional data

moduleauthor:: Derek Tucker <jdtuck@sandia.gov>

"""
import numpy as np
import fdasrsf.utility_functions as uf
from scipy.integrate import cumtrapz
import fdasrsf.fPCA as fpca
import fdasrsf.geometry as geo
import collections



