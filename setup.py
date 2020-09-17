import setuptools
import numpy
import sys, os
import platform
import findblas
from findblas.distutils import build_ext_with_blas
from distutils.core import setup
from distutils.core import Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Make sure I have the right Python version.
if sys.version_info[:2] < (3, 6):
    print(("fdasrsf requires Python 3.6 or newer. Python %d.%d detected" % sys.version_info[:2]))
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

gropt_src = []
for file in os.listdir("src/gropt/src/"):
    if file.endswith(".cpp"):
        gropt_src.append(os.path.join("src/gropt/src/", file))
gropt_src.insert(0,"src/optimum_reparam_Ng.pyx")
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.14'
class build_ext_subclass( build_ext_with_blas ):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args += ['/O2']
        else: # everything else that cares about following standards
            for e in self.extensions:
                if platform.system() == 'Darwin':
                    e.extra_compile_args += ['-stdlib=libc++']
                else:
                    e.extra_compile_args += ['-std=c++11']

        build_ext_with_blas.build_extensions(self)

extensions = [
	Extension(name="optimum_reparamN2",
	    sources=["src/optimum_reparamN2.pyx", "src/DynamicProgrammingQ2.c",
        "src/dp_grid.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="fpls_warp",
	    sources=["src/fpls_warp.pyx", "src/fpls_warp_grad.c", "src/misc_funcs.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="mlogit_warp",
	    sources=["src/mlogit_warp.pyx", "src/mlogit_warp_grad.c", "src/misc_funcs.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="ocmlogit_warp",
	    sources=["src/ocmlogit_warp.pyx", "src/ocmlogit_warp_grad.c", "src/misc_funcs.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
    Extension(name="oclogit_warp",
        sources=["src/oclogit_warp.pyx", "src/oclogit_warp_grad.c", "src/misc_funcs.c"],
        include_dirs=[numpy.get_include()],
        language="c"
    ),
    Extension(name="optimum_reparam_N",
        sources=["src/optimum_reparam_N.pyx", "src/DP.c"],
        include_dirs=[numpy.get_include()],
        language="c"
    ),
    Extension(name="optimum_reparam_Ng",
        sources=gropt_src,
        include_dirs=[numpy.get_include(), "src/gropt/incl/"],
        language="c++"
    ),
    Extension(name="cbayesian",
        sources=["src/cbayesian.pyx", "src/bayesian.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++"
    ),
]


setup(
    cmdclass={'build_ext': build_ext_subclass, 'build_docs': build_docs},
	ext_modules=extensions,
    name='fdasrsf',
    version='2.1.6',
    packages=['fdasrsf'],
    url='http://research.tetonedge.net',
    license='LICENSE.txt',
    author='J. Derek Tucker',
    author_email='jdtuck@sandia.gov',
    scripts=['bin/ex_srsf_align.py'],
    keywords=['functional data analysis'],
    description='functional data analysis using the square root slope framework',
    long_description=open('README.md', encoding="utf8").read(),
    data_files=[('share/man/man1', ['doc/build/man/fdasrsf.1'])],
    setup_requires=['findblas'],
    install_requires=[
        "Cython",
        "matplotlib",
        "numpy",
        "scipy",
        "joblib",
        "patsy",
        "tqdm",
        "findblas",
        "six",
        "pyparsing",
    ],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)
