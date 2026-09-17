
set -xe

PROJECT_DIR="$1"

printenv

python -m pip install delvewheel wheel mkl-devel

# Locate MKL from the installed mkl-devel wheel directly.  This used to go
# through findblas, but findblas only scans for a file *named* like a BLAS and
# nothing else here depends on it now: setup.py looks for MKL in this same
# <prefix>/Library/lib itself (see _windows_blas_dirs).
prefix=$(python -c "import sys, os; print(os.path.join(sys.prefix, 'Library'))")
lib_loc="$prefix/lib"
include_loc="$prefix/include"

test -d "$lib_loc" || { echo "MKL lib dir not found: $lib_loc"; exit 1; }
test -d "$include_loc" || { echo "MKL include dir not found: $include_loc"; exit 1; }
ls "$lib_loc"/mkl_rt* >/dev/null 2>&1 || { echo "no mkl_rt import library in $lib_loc"; ls "$lib_loc"; exit 1; }

libdir="C:\\WINDOWS"
cp -r $lib_loc/* $libdir
cp -r $include_loc/* $libdir
