import numpy
import sys
from distutils.core import setup
from distutils.core import Command
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Make sure I have the right Python version.
if sys.version_info[:2] < (2, 6):
    print("fdasrsf requires Python 2.6 or newer. Python %d.%d detected" % sys.version_info[:2])
    sys.exit(-1)


class clean(Command):
    """Cleans *.pyc and debian trashs, so you should get the same copy as
    is in the VCS.
    """

    description = "remove build files"
    user_options = [("all", "a", "the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os

        os.system("py.cleanup")
        os.system("rm -f python-build-stamp-2.4")
        os.system("rm -f MANIFEST")
        os.system("rm -rf build")
        os.system("rm -rf dist")
        os.system("rm -rf doc/_build")


ext_modules = [Extension(
    name="optimum_reparamN",
    sources=["src/optimum_reparamN.pyx", "src/DynamicProgrammingQ2.c", "src/dp_grid.c"],
    include_dirs=[numpy.get_include()], # .../site-packages/numpy/core/include
    language="c",
    # libraries=
    # extra_compile_args = "...".split(),
    # extra_link_args = "...".split()
)]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,

    name='fdasrsf',
    version='1.0.1',
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
