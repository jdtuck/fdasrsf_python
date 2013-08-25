import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

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
    version='1.0.0',
    package_dir={'fdasrsf': ''},
    packages=['fdasrsf'],
    url='http://stat.fsu.edu/~dtucker/research.html',
    license='LICENSE.txt',
    author='J. Derek Tucker',
    author_email='dtucker@stat.fsu.edu',
    scripts=['bin/ex_srsf_align.py'],
    keywords=['functional data analysis'],
    description=('functional data analysis using the square root slope framework'),
    long_description=open('README.txt').read(),
    requires=[
        "Cython",
        "matplotlib",
        "numpy",
        "scipy",
    ]
)
