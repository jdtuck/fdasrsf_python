from cffi import FFI
import os

ffibuilder = FFI()

with open(os.path.join(os.path.dirname(__file__), "DP.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source("_DP",
    '#include "DP.h"',
    sources=["DP.c"],
)

ffibuilder.compile()