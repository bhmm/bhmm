import numpy
import ctypes
cimport numpy

cdef extern from "_discrete.h":
    void _update_pout(int* obs, double* weights, int T, int N, int M, double* pout)

def update_pout(obs, weights, pout, dtype=numpy.float32):
    T = len(obs)
    N, M = numpy.shape(pout)

    # cdef numpy.ndarray[int, ndim=1, mode="c"] obs
    obs = numpy.ascontiguousarray(obs, dtype=ctypes.c_int)

    # pointers to arrays
    if dtype == numpy.float64:
        pobs     = <int*>    numpy.PyArray_DATA(obs)
        pweights = <double*> numpy.PyArray_DATA(weights)
        ppout    = <double*> numpy.PyArray_DATA(pout)
        # call
        _update_pout(pobs, pweights, T, N, M, ppout)
    else:
        raise TypeError

