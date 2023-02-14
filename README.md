[![Build Status](https://travis-ci.com/jdtuck/fdasrsf_python.svg?branch=master)](https://travis-ci.com/github/jdtuck/fdasrsf_python)
[![codecov](https://codecov.io/gh/jdtuck/fdasrsf_python/branch/master/graph/badge.svg)](https://codecov.io/gh/jdtuck/fdasrsf_python)
[![Build status](https://img.shields.io/appveyor/ci/jdtuck/fdasrsf-python.svg?style=flat-square&label=windows)](https://ci.appveyor.com/project/jdtuck/fdasrsf-python/branch/master)
[![Documentation Status](https://readthedocs.org/projects/fdasrsf-python/badge/?version=latest)](https://fdasrsf-python.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/fdasrsf.svg)](https://badge.fury.io/py/fdasrsf)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdtuck/fdasrsf_python/master?filepath=%2Fnotebooks)
[![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/tetonedge/fdasrsf)](https://hub.docker.com/r/tetonedge/fdasrsf)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/fdasrsf/badges/version.svg)](https://anaconda.org/conda-forge/fdasrsf) [![Join the chat at https://gitter.im/fdasrsf_python/community](https://badges.gitter.im/fdasrsf_python/community.svg)](https://gitter.im/fdasrsf_python/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

fdasrsf
=======

A python package for functional data analysis using the square root
slope framework and curves using the square root velocity framework
which performs pair-wise and group-wise alignment as well as modeling
using functional component analysis and regression. 

### Installation
------------------------------------------------------------------------------
v2.4.0 is on pip and can be installed using
> `pip install fdasrsf`

or conda

> `conda install -c conda-forge fdasrsf`

To install the most up to date version on github
> `pip install -e .`

please see [requirements](requirements.txt) for a list of packages `fdasrsf`
depends on

------------------------------------------------------------------------------

### Documentation
The documentation is available at
[fdasrsf-python.readthedocs.io/en/latest](https://fdasrsf-python.readthedocs.io/en/latest/), which
includes detailed information of the different modules, classes and methods of
the package, along with several examples showing different functionalities.

### Contributions
All contributions are welcome. You can help this project be better by reporting issues, bugs, 
or forking the repo and creating a pull request.

### License
The package is licensed under the BSD 3-Clause License. A copy of the
[license](LICENSE.txt) can be found along with the code.

### References
See references below on methods implemented in this package, some of the papers can be
found at this [website](http://research.tetonedge.net)

Tucker, J. D. 2014, Functional Component Analysis and Regression using Elastic
Methods. Ph.D. Thesis, Florida State University.

Robinson, D. T. 2012, Function Data Analysis and Partial Shape Matching in the
Square Root Velocity Framework. Ph.D. Thesis, Florida State University.

Huang, W. 2014, Optimization Algorithms on Riemannian Manifolds with
Applications. Ph.D. Thesis, Florida State University.

Srivastava, A., Wu, W., Kurtek, S., Klassen, E. and Marron, J. S. (2011).
Registration of Functional Data Using Fisher-Rao Metric. arXiv:1103.3817v2
[math.ST].

Tucker, J. D., Wu, W. and Srivastava, A. (2013). Generative models for
functional data using phase and amplitude separation. Computational Statistics
and Data Analysis 61, 50-66.

J. D. Tucker, W. Wu, and A. Srivastava, "Phase-Amplitude Separation of
Proteomics Data Using Extended Fisher-Rao Metric," Electronic Journal of
Statistics, Vol 8, no. 2. pp 1724-1733, 2014.

J. D. Tucker, W. Wu, and A. Srivastava, "Analysis of signals under compositional
noise With applications to SONAR data," IEEE Journal of Oceanic Engineering, Vol
29, no. 2. pp 318-330, Apr 2014.

Srivastava, A., Klassen, E., Joshi, S., Jermyn, I., (2011). Shape analysis of
elastic curves in euclidean spaces. Pattern Analysis and Machine Intelligence,
IEEE Transactions on 33 (7), 1415-1428.

S. Kurtek, A. Srivastava, and W. Wu. Signal estimation under random
time-warpings and nonlinear signal alignment. In Proceedings of Neural
Information Processing Systems (NIPS), 2011.

Wen Huang, Kyle A. Gallivan, Anuj Srivastava, Pierre-Antoine Absil. "Riemannian
Optimization for Elastic Shape Analysis", Short version, The 21st International
Symposium on Mathematical Theory of Networks and Systems (MTNS 2014).

Cheng, W., Dryden, I. L., and Huang, X. (2016). Bayesian registration of functions
and curves. Bayesian Analysis, 11(2), 447-475.

W. Xie, S. Kurtek, K. Bharath, and Y. Sun, A geometric approach to visualization
of variability in functional data, Journal of American Statistical Association 112
(2017), pp. 979-993.

Lu, Y., R. Herbei, and S. Kurtek, 2017: Bayesian registration of functions with a Gaussian process prior. Journal of
Computational and Graphical Statistics, 26, no. 4, 894–904.

Lee, S. and S. Jung, 2017: Combined analysis of amplitude and phase variations in functional data. arXiv:1603.01775 [stat.ME], 1–21.

J. D. Tucker, J. R. Lewis, and A. Srivastava, “Elastic Functional Principal Component Regression,” Statistical Analysis and Data Mining, vol. 12, no. 2, pp. 101-115, 2019.

J. D. Tucker, J. R. Lewis, C. King, and S. Kurtek, “A Geometric Approach for Computing Tolerance Bounds for Elastic Functional Data,” Journal of Applied Statistics, 10.1080/02664763.2019.1645818, 2019.

T. Harris, J. D. Tucker, B. Li, and L. Shand, "Elastic depths for detecting shape anomalies in functional data," Technometrics, 10.1080/00401706.2020.1811156, 2020.

M. K. Ahn, J. D. Tucker, W. Wu, and A. Srivastava. “Regression Models Using Shapes of Functions as Predictors” Computational Statistics and Data Analysis, 10.1016/j.csda.2020.107017, 2020. 

J. D. Tucker, L. Shand, and K. Chowdhary. “Multimodal Bayesian Registration of Noisy Functions using Hamiltonian Monte Carlo”, Computational Statistics and Data Analysis, accepted, 2021.

Q. Xie, S. Kurtek, E. Klassen, G. E. Christensen and A. Srivastava. Metric-based pairwise and multiple image registration. IEEE European Conference on Computer Vision (ECCV), September, 2014

X. Zhang, S. Kurtek, O. Chkrebtii, and J. D. Tucker, “Elastic k-means clustering of functional data 
   for posterior exploration, with an application to inference on acute respiratory infection dynamics”, 
   arXiv:2011.12397 [stat.ME], 2020.

