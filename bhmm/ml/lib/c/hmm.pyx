r"""Cython implementation of HMM forward backward functions.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""
import numpy
cimport numpy

cdef extern from "_hmm.h":
    double _forward(double * alpha, double * scaling, const double *A, const double *pobs, const double *pi, const int N, const int T)

cdef extern from "_hmm.h":
    void _backward(double *beta, double *scaling, const double *A, const double *pobs, const int N, const int T)

cdef extern from "_hmm.h":
    void _computeGamma(double *gamma, const double *alpha, const double *beta, const int T, const int N)

cdef extern from "_hmm.h":
    void _compute_transition_counts(double *transition_counts, const double *A, const double *pobs, const double *alpha, const double *beta, int N, int T)

cdef extern from "_hmm.h":
    void _compute_viterbi(int *path, const double *A, const double *pobs, const double *pi, int N, int T)

def forward(A, pobs, pi, dtype=numpy.float32):
    print "in forward pyx"
    N = A.shape[0]
    T = pobs.shape[0]
    # prepare alpha array
    cdef numpy.ndarray[double, ndim=2, mode="c"] alpha   = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
    # prepare scaling array
    cdef numpy.ndarray[double, ndim=1, mode="c"] scaling = numpy.zeros( (T), dtype=numpy.double, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        palpha = <double*> numpy.PyArray_DATA(alpha)
        pscaling = <double*> numpy.PyArray_DATA(scaling)
        pA = <double*> numpy.PyArray_DATA(A)
        ppobs = <double*> numpy.PyArray_DATA(pobs)
        ppi = <double*> numpy.PyArray_DATA(pi)
        # call
        logprob = _forward(palpha, pscaling, pA, ppobs, ppi, N, T)
        print "forward done"
        print "logprob: ",logprob
        print "alpha", alpha
        print "scaling",scaling
        return logprob, alpha, scaling
    else:
        raise ValueError

def backward(A, pobs, dtype=numpy.float32):
    print "in backward pyx"
    N = A.shape[0]
    T = pobs.shape[0]
    # prepare alpha array
    cdef numpy.ndarray[double, ndim=2, mode="c"] beta    = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
    # prepare scaling array
    cdef numpy.ndarray[double, ndim=1, mode="c"] scaling = numpy.zeros( (T), dtype=numpy.double, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        pbeta    = <double*> numpy.PyArray_DATA(beta)
        pscaling = <double*> numpy.PyArray_DATA(scaling)
        pA       = <double*> numpy.PyArray_DATA(A)
        ppobs    = <double*> numpy.PyArray_DATA(pobs)
        # call
        _backward(pbeta, pscaling, pA, ppobs, N, T)
        print "backward done"
        print "beta", beta
        return beta
    else:
        raise ValueError


def state_probabilities(alpha, beta, dtype=numpy.float32):
    print "in gamma pyx"
    T = alpha.shape[0]
    N = alpha.shape[1]
    # prepare alpha array
    cdef numpy.ndarray[double, ndim=2, mode="c"] gamma = numpy.zeros( (T,N), dtype=numpy.double, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        pgamma   = <double*> numpy.PyArray_DATA(gamma)
        palpha   = <double*> numpy.PyArray_DATA(alpha)
        pbeta    = <double*> numpy.PyArray_DATA(beta)
        # call
        _computeGamma(pgamma, palpha, pbeta, N, T)
        print "gamma done"
        print "gamma", gamma
        return gamma
    else:
        raise ValueError

def transition_counts(alpha, beta, A, pobs, dtype=numpy.float32):
    print "in transition count pyx"
    N = A.shape[0]
    T = alpha.shape[0]
    # prepare alpha array
    cdef numpy.ndarray[double, ndim=2, mode="c"] C = numpy.zeros( (N,N), dtype=numpy.double, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        pC = <double*> numpy.PyArray_DATA(C)
        pA = <double*> numpy.PyArray_DATA(A)
        ppobs = <double*> numpy.PyArray_DATA(pobs)
        palpha   = <double*> numpy.PyArray_DATA(alpha)
        pbeta   = <double*> numpy.PyArray_DATA(beta)
        # call
        _compute_transition_counts(pC, pA, ppobs, palpha, pbeta, N, T)
        print "transition counts done"
        print "transition counts", C
        return C
    else:
        raise ValueError

def viterbi(A, pobs, pi, dtype=numpy.float32):
    print "in viterbi pyx"
    N = A.shape[0]
    T = pobs.shape[0]
    # prepare path array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path = numpy.zeros( (T), dtype=numpy.int32, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        ppath = <int*>    numpy.PyArray_DATA(path)
        pA    = <double*> numpy.PyArray_DATA(A)
        ppobs = <double*> numpy.PyArray_DATA(pobs)
        ppi   = <double*> numpy.PyArray_DATA(pi)
        # call
        _compute_viterbi(ppath, pA, ppobs, ppi, N, T)
        print "viterbi done"
        print "path", path
        return path
    else:
        raise ValueError
