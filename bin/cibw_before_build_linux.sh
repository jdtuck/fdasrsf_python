set -xe


NIGHTLY_FLAG=""

if [ "$#" -eq 1 ]; then
    PROJECT_DIR="$1"
elif [ "$#" -eq 2 ] && [ "$1" = "--nightly" ]; then
    NIGHTLY_FLAG="--nightly"
    PROJECT_DIR="$2"
else
    echo "Usage: $0 [--nightly] <project_dir>"
    exit 1
fi

printenv
# Update license



# Install OpenBLAS
python -m pip install -r requirements_openblas.txt
include_dir=$(python -c "import scipy_openblas32; print(scipy_openblas32.get_include_dir())")  
lib_dir=$(python -c "import scipy_openblas32; print(scipy_openblas32.get_lib_dir())")  
cp -r $lib_dir/* /usr/local/lib
cp $include_dir/* /usr/local/include
