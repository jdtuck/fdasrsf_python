
set -xe

PROJECT_DIR="$1"

printenv

python -m pip install delvewheel wheel mkl-devel

# Locate MKL from the installed mkl-devel wheel directly.  This used to go
# through findblas, but findblas only scans for a file *named* like a BLAS and
# nothing else here depends on it now: setup.py looks for MKL in this same
# <prefix>/Library/lib itself (see _windows_blas_dirs).
#
# Ask for a forward-slash prefix.  sys.prefix is a native Windows path, and
# os.path.join would give us 'C:\...\venv\Library' -- a name this shell can pass
# to a program but cannot glob, because in a POSIX shell a backslash quotes the
# character after it, so the '*' in an unquoted "$lib_loc"/* stays literal and
# 'cp' fails with "cannot stat '...Library/lib/*'".
prefix=$(python -c "import sys; print(sys.prefix.replace('\\\\', '/') + '/Library')")
lib_loc="$prefix/lib"
include_loc="$prefix/include"

test -d "$lib_loc" || { echo "MKL lib dir not found: $lib_loc"; exit 1; }
test -d "$include_loc" || { echo "MKL include dir not found: $include_loc"; exit 1; }
ls "$lib_loc"/mkl_rt* >/dev/null 2>&1 || { echo "no mkl_rt import library in $lib_loc"; ls "$lib_loc"; exit 1; }

libdir="/c/Windows"
cp -r "$lib_loc"/* "$libdir"
cp -r "$include_loc"/* "$libdir"
