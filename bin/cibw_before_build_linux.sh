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

PLATFORM=$(PYTHONPATH=bin python -c "import openblas_support; print(openblas_support.get_plat())")

printenv
# Update license

# Install Openblas
basedir=$(python bin/openblas_support.py $NIGHTLY_FLAG)
cp -r $basedir/lib/* /usr/local/lib
cp $basedir/include/* /usr/local/include