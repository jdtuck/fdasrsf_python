set -xe

PROJECT_DIR="$1"

printenv

# Install OpenBLAS
python -m pip install -r bin/requirements_openblas.txt
python -c "import scipy_openblas32; print(scipy_openblas32.get_pkg_config())" > $PROJECT_DIR/scipy-openblas.pc

# delvewheel is the equivalent of delocate/auditwheel for windows.
python -m pip install delvewheel wheel