import setuptools
import numpy
import glob
import re
import subprocess
import sys, os
import platform
from setuptools import setup
from setuptools import Command

from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import dist
from sysconfig import get_config_var
from packaging.version import parse as LooseVersion


def _windows_blas_dirs():
    """Directories a Windows build may find a BLAS import library in.

    MSVC, unlike the Unix linkers, has no default place to look for a BLAS, and
    a stock Windows box has none.  Both CI and the install docs get one from the
    'mkl-devel' wheel, which unpacks into the same <prefix>/Library layout conda
    uses.  pip's isolated build environment is the same interpreter with a
    rewritten sys.path rather than a separate venv, so sys.prefix still points
    at the environment mkl-devel was installed into; sys.base_prefix covers a
    venv whose base holds it instead.  LIB and C:\\Windows are what
    bin/cibw_before_build_win.sh stages MKL into for the wheel builds.
    """
    dirs = [
        os.path.join(sys.prefix, "Library", "lib"),
        os.path.join(sys.base_prefix, "Library", "lib"),
    ]
    dirs += [d for d in os.environ.get("LIB", "").split(os.pathsep) if d]
    dirs.append(os.environ.get("SystemRoot", r"C:\Windows"))

    seen = set()
    return [d for d in dirs if d not in seen and not seen.add(d)]


def _find_blas(preferred=None, preferred_dir=None):
    """Locate a BLAS to link against, as a (library name, directory) pair.

    'directory' is None when the linker's own search path already finds the
    library, which is the case everywhere except Windows.  'preferred' and
    'preferred_dir' are the name and directory asked for via FDASRSF_BLAS_LIB /
    FDASRSF_BLAS_DIR.
    """
    if sys.platform != "win32":
        # 'blas' is provided by conda-forge (libblas), by Debian/Ubuntu (the
        # libblas.so alternative that libopenblas-dev installs) and by the macOS
        # SDK (libblas.tbd, i.e. Accelerate), and the linker finds it unaided.
        return preferred or "blas", preferred_dir

    # Most-specific first: MKL is what the Windows instructions install, and
    # matching on a prefix picks up the versioned import libraries (mkl_rt.2.lib)
    # that newer MKL releases ship instead of a bare mkl_rt.lib.
    if preferred:
        names = [preferred]
    else:
        names = ["mkl_rt", "openblas", "scipy_openblas", "blas"]

    dirs = _windows_blas_dirs()
    if preferred_dir:
        dirs.insert(0, preferred_dir)

    for directory in dirs:
        for name in names:
            exact = os.path.join(directory, name + ".lib")
            if os.path.exists(exact):
                return name, directory
            hits = sorted(glob.glob(os.path.join(directory, name + "*.lib")))
            if hits:
                return os.path.splitext(os.path.basename(hits[0]))[0], directory

    # Nothing found; name the preferred library anyway so the linker reports a
    # missing library rather than a wall of unresolved BLAS symbols.
    return names[0], preferred_dir


def _find_lapack(blas_lib, blas_dir):
    """The LAPACK import library MSVC needs alongside 'blas_lib', if any.

    Returns a list of extra library names to put on the link line -- empty
    everywhere except a Windows build whose BLAS ships LAPACK separately.

    rbfgs.cpp calls no LAPACK (it compiles clean under -DARMA_DONT_USE_LAPACK),
    but MSVC still emits references to it.  Armadillo's auxlib dispatches on
    element type inside a single function body, so a call it cannot use is a
    dead branch rather than an uninstantiated template; clang and gcc fold those
    branches away at -O2 and the references vanish, MSVC keeps them.  The
    result is a link that needs gesv_, posv_ and gesdd_ (each in s/d/c/z form)
    resolved even though nothing will ever call them.

    Whether that costs a second '-l' depends on how the BLAS is packaged:

      * MKL and OpenBLAS put LAPACK in the same import library as BLAS, so
        naming the BLAS is enough -- this is what the Windows install
        instructions and the cibuildwheel jobs provide, which is why the gap
        went unnoticed;
      * conda-forge splits them, one DLL and one import library each
        (blas.lib -> libblas.dll, lapack.lib -> liblapack.dll).  There, linking
        only the BLAS fails outright:

            rbfgs.obj : error LNK2001: unresolved external symbol dgesdd_
            ... fatal error LNK1120: 10 unresolved externals

    So look for a matching LAPACK next to the BLAS and add it when one exists.
    FDASRSF_LAPACK_LIB names it explicitly for a layout not covered here.
    """
    if sys.platform != "win32":
        return []

    requested = os.environ.get("FDASRSF_LAPACK_LIB")
    if requested is not None:
        # An empty value is a deliberate "this BLAS needs no separate LAPACK".
        return [requested] if requested else []

    # A combined BLAS+LAPACK needs nothing added; naming its LAPACK a second
    # time would be harmless but misleading in the build log.
    if re.match(r"mkl|openblas|scipy_openblas", blas_lib, re.I):
        return []

    dirs = _windows_blas_dirs()
    if blas_dir:
        dirs.insert(0, blas_dir)

    for directory in dirs:
        for name in ("lapack", "liblapack"):
            if os.path.exists(os.path.join(directory, name + ".lib")):
                return [name]

    return []


def blas_link_args():
    """Link settings for extensions that reference BLAS directly.

    'crbfgs' is the only one: rbfgs.cpp uses Armadillo's norm()/dot(), and the
    vendored armadillo_bits/config.hpp enables ARMA_USE_BLAS with
    ARMA_USE_WRAPPER left off, so its objects carry direct references to the
    Fortran BLAS symbols dnrm2_, ddot_, dasum_ and dgemv_.  (bayesian.cpp and
    UnitSquareImage.cpp reference none, so they need nothing here.)

    This used to be delegated to findblas's 'build_ext_with_blas', but findblas
    works by probing well-known directories for a file *named* like a BLAS --
    which is why bin/cibw_before_build_*.sh stage OpenBLAS into /usr/local/lib
    and alias libscipy_openblas to libopenblas.  Nothing stages /usr/local in a
    conda-forge build, so discovery there produced no '-l' flag at all, and
    because ELF permits unresolved symbols the result was a module that linked
    cleanly and then failed at dlopen:

        ImportError: ...crbfgs...so: undefined symbol: dnrm2_

    Mach-O rejects undefined symbols by default, which is why macOS builds and
    the local test suite never showed it.

    FDASRSF_BLAS_LIB / FDASRSF_BLAS_DIR override the library name and the
    directory it is looked for in; both are optional, and the defaults are
    whatever the platform provides (see _find_blas).  The cibuildwheel jobs set
    the name because they stage scipy-openblas32, whose library is
    'libscipy_openblas' rather than 'libblas'; see the [tool.cibuildwheel.*]
    environment tables in pyproject.toml.

    A Windows build may also need LAPACK named separately even though this code
    calls none; see _find_lapack.
    """
    requested = os.environ.get("FDASRSF_BLAS_LIB")
    requested_dir = os.environ.get("FDASRSF_BLAS_DIR")

    # Search even when a name was requested: on Windows that is how a versioned
    # import library ('mkl_rt.2.lib' for FDASRSF_BLAS_LIB=mkl_rt) is resolved.
    lib, lib_dir = _find_blas(requested, requested_dir)

    return {
        "libraries": [lib] + _find_lapack(lib, lib_dir),
        "library_dirs": [lib_dir] if lib_dir else [],
        "include_dirs": [numpy.get_include()],
    }

# Make sure I have the right Python version.
if sys.version_info[:2] < (3, 10):
    print(
        (
            "fdasrsf requires Python 3.10 or newer. Python %d.%d detected"
            % sys.version_info[:2]
        )
    )
    sys.exit(-1)


#: BLAS routines rbfgs.cpp reaches through Armadillo's norm()/dot(); if the
#: module can satisfy these it will import, however that was arranged.
BLAS_SYMBOLS = ("dnrm2_", "ddot_", "dasum_", "dgemv_")


class build_ext_checked(build_ext):
    """build_ext that refuses to emit a BLAS extension linked without BLAS.

    ELF shared objects may legally contain undefined symbols, so a botched BLAS
    link only shows up much later as an ImportError at dlopen time -- which is
    how a 'crbfgs' missing dnrm2_ got all the way into a conda-forge build.
    Check the freshly linked objects here so the failure lands on the build.

    The check asks the question dlopen will ask: is every BLAS symbol either
    defined in the module or backed by a DT_NEEDED library that can supply it?
    Both are legitimate outcomes, and which one you get depends on the BLAS that
    was found rather than on anything this project controls:

      * a shared BLAS leaves the symbols undefined and records a DT_NEEDED --
        this is the usual conda-forge / distro case;
      * a static BLAS (a '.a', which is what the scipy-openblas32 wheel ships
        for Linux) is pulled straight into the module, so the symbols come out
        *defined* and there is deliberately no DT_NEEDED for it.

    An earlier version of this check demanded a DT_NEEDED unconditionally and so
    rejected the perfectly good statically linked wheel builds.

    Best-effort by design: if the platform is not ELF, or no symbol reader is
    available, this stays quiet rather than blocking the build.
    """

    #: extensions whose objects reference BLAS and so must be able to resolve it
    needs_blas = ("crbfgs",)

    def run(self):
        super().run()

        if sys.platform in ("darwin", "win32"):
            # Both linkers reject undefined symbols outright, so a missing BLAS
            # link fails at build time already.
            return

        for ext in self.extensions:
            if ext.name not in self.needs_blas:
                continue
            path = self.get_ext_fullpath(ext.name)
            if not os.path.exists(path):
                continue
            self._check_blas_resolvable(path)

    def _check_blas_resolvable(self, path):
        # Look in both symbol tables: a static BLAS built with hidden visibility
        # is defined in .symtab without appearing in .dynsym, and a stripped
        # module has only .dynsym.  Missing either one is not an error here.
        defined = ""
        for argv in (["nm", "-D", "--defined-only", path],
                     ["nm", "--defined-only", path]):
            try:
                defined += subprocess.run(
                    argv, capture_output=True, text=True, check=True
                ).stdout
            except (OSError, subprocess.CalledProcessError):
                continue

        if not defined:
            return  # no usable nm; skip rather than fail the build

        # Statically linked BLAS: the routines are part of the module already.
        if any(re.search(r"\b%s\b" % s, defined) for s in BLAS_SYMBOLS):
            return

        try:
            dyn = subprocess.run(
                ["readelf", "-d", path],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return  # no readelf; skip rather than fail the build

        # Otherwise they must come from a shared library at load time.
        needed = re.findall(r"NEEDED\).*?\[([^\]]+)\]", dyn)
        if not any(re.search(r"blas|lapack|mkl|accelerate", n, re.I) for n in needed):
            raise SystemExit(
                "%s references BLAS but cannot resolve it: the BLAS routines are "
                "neither linked into the module nor provided by a library it "
                "depends on (DT_NEEDED: %s).\n"
                "It would fail at import with 'undefined symbol: dnrm2_'. "
                "Make sure a BLAS is on the linker search path, or point at one "
                "with FDASRSF_BLAS_LIB / FDASRSF_BLAS_DIR."
                % (os.path.basename(path), ", ".join(needed) or "none")
            )


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
    # crbfgs is the one extension whose objects reference BLAS symbols directly
    # (see blas_link_args); it must be linked against BLAS explicitly.
    Extension(
        name="crbfgs",
        sources=["src/crbfgs.pyx", "src/rbfgs.cpp"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        **blas_link_args(),
    ),
    Extension(
        name="cimage",
        sources=["src/imagecpp.pyx", "src/UnitSquareImage.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),
]


setup(
    cmdclass={"build_ext": build_ext_checked, "build_docs": build_docs},
    ext_modules=extensions,
    cffi_modules=["src/dp_build.py:ffibuilder"],
    name="fdasrsf",
    version="2.7.2",
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
