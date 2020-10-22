from cffi import FFI
import os

ffibuilder = FFI()

with open(os.path.join(os.path.dirname(__file__), "DP.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source("_DP",
    '#include "DP.h"',
    sources=["src/DP.c"],
    include_dirs=["src/"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)