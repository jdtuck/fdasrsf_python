import numpy
import sys
from distutils.core import setup
from distutils.core import Command
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Make sure I have the right Python version.
if sys.version_info[:2] < (2, 6):
    print(("fdasrsf requires Python 2.6 or newer. Python %d.%d detected" % sys.version_info[:2]))
    sys.exit(-1)


class build_docs(Command):
    """Builds the documentation
    """

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
        os.system("sphinx-build -b man doc/source doc/build/man")
        os.chdir("doc/build/latex")
        os.system("latexmk -pdf fdasrsf.tex")
        os.chdir("../../../")


ext_modules_dp = Extension(
    name="optimum_reparamN",
    sources=["src/optimum_reparamN.pyx", "src/DynamicProgrammingQ2.c", "src/dp_grid.c"],
    include_dirs=[numpy.get_include()], # .../site-packages/numpy/core/include
    language="c",
    # libraries=
    # extra_compile_args = "...".split(),
    # extra_link_args = "...".split()
)

ext_modules_pls = Extension(
    name="fpls_warp",
    sources=["src/fpls_warp.pyx", "src/fpls_warp_grad.c", "src/misc_funcs.c"],
    include_dirs=[numpy.get_include()], # .../site-packages/numpy/core/include
    language="c",
    # libraries=
    # extra_compile_args = "...".split(),
    # extra_link_args = "...".split()
)

ext_modules_mlogit = Extension(
    name="mlogit_warp",
    sources=["src/mlogit_warp.pyx", "src/mlogit_warp_grad.c", "src/misc_funcs.c"],
    include_dirs=[numpy.get_include()], # .../site-packages/numpy/core/include
    language="c",
    # libraries=
    # extra_compile_args = "...".split(),
    # extra_link_args = "...".split()
)

setup(
    cmdclass={'build_ext': build_ext, 'build_docs': build_docs},
    ext_modules=[ext_modules_dp, ext_modules_pls, ext_modules_mlogit],

    name='fdasrsf',
    version='1.2.0',
    packages=['fdasrsf'],
    url='http://stat.fsu.edu/~dtucker/research.html',
    license='LICENSE.txt',
    author='J. Derek Tucker',
    author_email='dtucker@stat.fsu.edu',
    scripts=['bin/ex_srsf_align.py'],
    keywords=['functional data analysis'],
    description='functional data analysis using the square root slope framework',
    long_description=open('README.txt').read(),
    data_files=[('share/man/man1', ['doc/build/man/fdasrsf.1'])],
    requires=[
        "Cython",
        "matplotlib",
        "numpy",
        "scipy",
        "joblib",
        "patsy",
    ],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ]
)
