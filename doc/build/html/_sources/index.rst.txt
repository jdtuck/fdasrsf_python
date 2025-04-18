.. fdasrsf documentation master file, created by
   sphinx-quickstart on Tue Aug 20 21:12:55 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../artwork/logo.png
  :width: 200px
  :align: right

Welcome to fdasrsf's documentation!
===================================
A python package for functional data analysis using the square root
slope framework and curves using the square root velocity framework
which performs pair-wise and group-wise alignment as well as modeling
using functional component analysis and regression.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   
   user_guide.rst

.. toctree::
   :maxdepth: 1
   :titlesonly:
   
   api.rst

Installation
=============
Currently, *fdasrsf* is available in Python versions above 3.10, regardless of the
platform.
The stable version can be installed via
`PyPI <https://pypi.org/project/fdasrsf/>`_:

.. code-block:: bash

   pip install fdasrsf

It is also available from conda-forge:

.. code-block:: bash

    conda install -c conda-forge fdasrsf

It is possible to install the latest version of the package, available in
the develop branch, by cloning this repository and doing a manual installation.

.. code-block:: bash

   git clone https://github.com/jdtuck/fdasrsf_python.git
   pip install ./fdasrsf_python


In this type of installation make sure that your default Python version is
currently supported, or change the python and pip commands by specifying a
version, such as python3.8.

How do I start?
===============
If you want a quick overview of the package, we recommend you to look at
the example notebooks in the :doc:`Users Guide <user_guide>`

Contributions
=============
All contributions are welcome. You can help this project grow in multiple ways,
from creating an issue, reporting an improvement or a bug, to doing a
repository fork and creating a pull request to the development branch.

License
=======
The package is licensed under the BSD 3-Clause License. A copy of the
`license <https://github.com/jdtuck/fdasrsf_python/LICENSE.txt>`_
can be found along with the code or in the project page.

References
==========

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

   Lu, Y., R. Herbei, and S. Kurtek, 2017: Bayesian registration of functions with a 
   Gaussian process prior. Journal of Computational and Graphical Statistics, 26, no. 4, 894–904.

   Lee, S. and S. Jung, 2017: Combined analysis of amplitude and phase variations in 
   functional data. arXiv:1603.01775 [stat.ME], 1–21.

   J. D. Tucker, J. R. Lewis, and A. Srivastava, “Elastic Functional Principal Component 
   Regression,” Statistical Analysis and Data Mining, vol. 12, no. 2, pp. 101-115, 2019.

   J. D. Tucker, J. R. Lewis, C. King, and S. Kurtek, “A Geometric Approach for Computing 
   Tolerance Bounds for Elastic Functional Data,” Journal of Applied Statistics, 10.1080/02664763.2019.1645818, 2019.

   T. Harris, J. D. Tucker, B. Li, and L. Shand, "Elastic depths for detecting shape 
   anomalies in functional data," Technometrics, 10.1080/00401706.2020.1811156, 2020.

   M. K. Ahn, J. D. Tucker, W. Wu, and A. Srivastava. “Regression Models Using Shapes 
   of Functions as Predictors” Computational Statistics and Data Analysis, 10.1016/j.csda.2020.107017, 2020. 

   J. D. Tucker, L. Shand, and K. Chowdhary. “Multimodal Bayesian Registration of Noisy Functions 
   using Hamiltonian Monte Carlo”, Computational Statistics and Data Analysis, accepted, 2021.

   X. Zhang, S. Kurtek, O. Chkrebtii, and J. D. Tucker, “Elastic k-means clustering of functional data 
   for posterior exploration, with an application to inference on acute respiratory infection dynamics”, 
   arXiv:2011.12397 [stat.ME], 2020.

   Q. Xie, S. Kurtek, E. Klassen, G. E. Christensen and A. Srivastava. Metric-based pairwise and multiple image registration. IEEE European Conference on Computer Vision (ECCV), September, 2014

   J. D. Tucker and D. Yarger, “Elastic Functional Changepoint Detection of Climate Impacts from Localized Sources”, Envirometrics, 10.1002/env.2826, 2023.

   Yu, Q., Lu, X., and Marron, J. S. (2017), “Principal Nested Spheres for Time-Warped Functional Data Analysis,” Journal of Computational and Graphical Statistics, 26, 144–151.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

