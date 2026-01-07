set -xe

PROJECT_DIR="$1"

printenv

# Install OpenBLAS
python -m pip install -r bin/requirements_openblas.txt
python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > $PROJECT_DIR/scipy-openblas.pc

lib_loc=$(python -c"import scipy_openblas32; print(scipy_openblas32.get_lib_dir())")
include_loc=$(python -c"import scipy_openblas32; print(scipy_openblas32.get_include_dir())")

libdir=$(python -c"import sys; import os; print(os.path.join(sys.prefix, 'include'))")
includedir=$(python -c"import sys; import os; print(os.path.join(sys.prefix, 'Library', 'lib'))")

cp -r $lib_loc/* $libdir
cp $include_loc/* $includedir

cp $libdir/libscipy_openblas.dll $libdir/libopenblas.dll
cp $libdir/libscipy_openblas.lib $libdir/libopenblas.lib

# delvewheel is the equivalent of delocate/auditwheel for windows.

python -m pip install delvewheel wheel mkl-devel