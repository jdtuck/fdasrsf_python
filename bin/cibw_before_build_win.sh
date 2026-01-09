
set -xe

PROJECT_DIR="$1"

printenv

python -m pip install delvewheel wheel mkl-devel findblas
lib_loc=$(python -c"import findblas; blas_path, blas_file, incl_path, incl_file, flags = findblas.find_blas();print(blas_path)")
include_loc=$(python -c"import findblas; blas_path, blas_file, incl_path, incl_file, flags = findblas.find_blas();print(incl_path)")

libdir="C:\\WINDOWS"
cp -r $lib_loc/* $libdir
cp -r $include_loc/* $libdir
