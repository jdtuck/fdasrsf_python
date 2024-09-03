import setuptools
import numpy
import sys, os
import platform
from setuptools import setup
from setuptools import Command
from findblas.distutils import build_ext_with_blas
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import dist
from sysconfig import get_config_var
from packaging.version import parse as LooseVersion

# Make sure I have the right Python version.
if sys.version_info[:2] < (3, 10):
    print(
        (
            "fdasrsf requires Python 3.10 or newer. Python %d.%d detected"
            % sys.version_info[:2]
        )
    )
    sys.exit(-1)


class build_docs(Command):
    """Builds the documentation"""

    description = "builds the documentation"
    user_options = []

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os

        os.system("sphinx-build -b html doc/source doc/build/html")
        os.system("sphinx-build -b latex doc/source doc/build/latex")
        os.chdir("doc/build/latex")
        os.system("latexmk -pdf fdasrsf.tex")
        os.chdir("../../../")


if sys.platform == "darwin":
    mac_ver = str(LooseVersion(get_config_var("MACOSX_DEPLOYMENT_TARGET")))
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = mac_ver

extensions = [
    Extension(
        name="optimum_reparamN2",
        sources=[
            "src/optimum_reparamN2.pyx",
            "src/DynamicProgrammingQ2.c",
            "src/dp_grid.c",
            "src/dp_nbhd.c",
        ],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="fpls_warp",
        sources=["src/fpls_warp.pyx", "src/fpls_warp_grad.c", "src/misc_funcs.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="mlogit_warp",
        sources=["src/mlogit_warp.pyx", "src/mlogit_warp_grad.c", "src/misc_funcs.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="ocmlogit_warp",
        sources=[
            "src/ocmlogit_warp.pyx",
            "src/ocmlogit_warp_grad.c",
            "src/misc_funcs.c",
        ],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="oclogit_warp",
        sources=["src/oclogit_warp.pyx", "src/oclogit_warp_grad.c", "src/misc_funcs.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="optimum_reparam_N",
        sources=["src/optimum_reparam_N.pyx", "src/DP.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        name="cbayesian",
        sources=["src/cbayesian.pyx", "src/bayesian.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        name="crbfgs",
        sources=["src/crbfgs.pyx", "src/rbfgs.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        name="cimage",
        sources=["src/imagecpp.pyx", "src/UnitSquareImage.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),
]


setup(
    cmdclass={"build_ext": build_ext_with_blas, "build_docs": build_docs},
    ext_modules=extensions,
    cffi_modules=["fdasrsf/src/dp_build.py:ffibuilder"],
    name="fdasrsf",
    version="2.6.1",
    packages=["fdasrsf"],
    url="http://research.tetonedge.net",
    license="LICENSE.txt",
    author="J. Derek Tucker",
    author_email="jdtuck@sandia.gov",
    scripts=["bin/ex_srsf_align.py"],
    keywords=["functional data analysis"],
    description="functional data analysis using the square root slope framework",
    long_description=open("README.md", encoding="utf8").read(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
