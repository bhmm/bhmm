r"""Cython implementation of HMM forward backward functions.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""
import numpy
import ctypes
cimport numpy

cdef extern from "_hidden.h":
    double _forward(double * alpha, const double *A, const double *pobs, const double *pi, const int N, const int T)

cdef extern from "_hidden.h":
    void _backward(double *beta, const double *A, const double *pobs, const int N, const int T)

# cdef extern from "_hmm.h":
#     void _computeGamma(double *gamma, const double *alpha, const double *beta, const int T, const int N)
#
cdef extern from "_hidden.h":
    void _compute_transition_counts(double *transition_counts, const double *A, const double *pobs, const double *alpha, const double *beta, int N, int T)

cdef extern from "_hidden.h":
    void _compute_viterbi(int *path, const double *A, const double *pobs, const double *pi, int N, int T)

cdef extern from "_hidden.h":
    void _sample_path(int *path, const double *alpha, const double *A, const double *pobs, const int N, const int T)


def cdef_double_array(n1, n2):
    cdef numpy.ndarray[double, ndim=2, mode="c"] out = numpy.zeros( (n1,n2), dtype=numpy.double, order='C' )
    return out


def forward(A, pobs, pi, T=None, alpha_out=None, dtype=numpy.float32):
    # set T
    if (T is None):
        T = pobs.shape[0] # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise TypeError('T must be at most the length of pobs.')
    # set N
    N = A.shape[0]
    # prepare alpha array
    # initialize output if necessary
    #cdef numpy.ndarray[double, ndim=2, mode="c"] alpha = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
    if alpha_out is None:
        alpha = cdef_double_array(T,N)
    elif T > alpha_out.shape[0]:
        raise TypeError('alpha_out must at least have length T in order to fit trajectory.')
    else:
        alpha = alpha_out

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        palpha = <double*> numpy.PyArray_DATA(alpha)
        pA = <double*> numpy.PyArray_DATA(A)
        ppobs = <double*> numpy.PyArray_DATA(pobs)
        ppi = <double*> numpy.PyArray_DATA(pi)
        # call
        logprob = _forward(palpha, pA, ppobs, ppi, N, T)
        return logprob, alpha
    else:
        raise TypeError

def backward(A, pobs, T=None, beta_out=None, dtype=numpy.float32):
    # set T
    if (T is None):
        T = pobs.shape[0] # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = A.shape[0]
    #cdef numpy.ndarray[double, ndim=2, mode="c"] beta = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
    # prepare beta array
    if beta_out is None:
        beta = cdef_double_array(T,N)
        #cdef numpy.ndarray[double, ndim=2, mode="c"] beta_out = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
    elif T > beta_out.shape[0]:
        raise ValueError('beta_out must at least have length T in order to fit trajectory.')
    else:
        beta = beta_out

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        pbeta    = <double*> numpy.PyArray_DATA(beta)
        pA       = <double*> numpy.PyArray_DATA(A)
        ppobs    = <double*> numpy.PyArray_DATA(pobs)
        # call
        _backward(pbeta, pA, ppobs, N, T)
        return beta
    else:
        raise TypeError


# def state_probabilities(alpha, beta, gamma_out=None, dtype=numpy.float32):
#     T = alpha.shape[0]
#     N = alpha.shape[1]
#     # prepare gamma array
#     cdef numpy.ndarray[double, ndim=2, mode="c"] gamma = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
#     #if gamma_out is None:
#     #    gamma_out = cdef_double_array(2,(T,N))
#         #cdef numpy.ndarray[double, ndim=2, mode="c"] gamma_out = numpy.zeros( (T,N), dtype=numpy.double, order='C' )
#
#     #if dtype == numpy.float32:
#     #    return ext.forward32(A, pobs, pi)
#     if dtype == numpy.float64:
#         pgamma   = <double*> numpy.PyArray_DATA(gamma)
#         palpha   = <double*> numpy.PyArray_DATA(alpha)
#         pbeta    = <double*> numpy.PyArray_DATA(beta)
#         # call
#         _computeGamma(pgamma, palpha, pbeta, N, T)
#         return gamma
#     else:
#         raise ValueError
#
#
def transition_counts(alpha, beta, A, pobs, T = None, out = None, dtype=numpy.float32):
    # set T
    if (T is None):
        T = pobs.shape[0] # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = len(A)
    # prepare alpha array
    # output
    #cdef numpy.ndarray[double, ndim=2, mode="c"] C = numpy.zeros( (N,N), dtype=numpy.double, order='C' )
    if out is None:
        C = cdef_double_array(N,N)
    else:
        C = out
        #cdef numpy.ndarray[double, ndim=2, mode="c"] out = numpy.zeros( (N,N), dtype=numpy.double, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        pC     = <double*> numpy.PyArray_DATA(C)
        pA     = <double*> numpy.PyArray_DATA(A)
        ppobs  = <double*> numpy.PyArray_DATA(pobs)
        palpha = <double*> numpy.PyArray_DATA(alpha)
        pbeta  = <double*> numpy.PyArray_DATA(beta)
        # call
        _compute_transition_counts(pC, pA, ppobs, palpha, pbeta, N, T)
        return C
    else:
        raise TypeError


def viterbi(A, pobs, pi, dtype=numpy.float32):
    N = A.shape[0]
    T = pobs.shape[0]
    # prepare path array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path
    path = numpy.zeros( (T), dtype=ctypes.c_int, order='C' )

    #if dtype == numpy.float32:
    #    return ext.forward32(A, pobs, pi)
    if dtype == numpy.float64:
        ppath = <int*>    numpy.PyArray_DATA(path)
        pA    = <double*> numpy.PyArray_DATA(A)
        ppobs = <double*> numpy.PyArray_DATA(pobs)
        ppi   = <double*> numpy.PyArray_DATA(pi)
        # call
        _compute_viterbi(ppath, pA, ppobs, ppi, N, T)
        return path
    else:
        raise TypeError


def sample_path(alpha, A, pobs, T = None, dtype=numpy.float32):
    N = pobs.shape[1]
    # set T
    if (T is None):
        T = pobs.shape[0] # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0] or T > alpha.shape[0]:
        raise ValueError('T must be at most the length of pobs and alpha.')
    # prepare path array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path
    path = numpy.zeros( (T), dtype=ctypes.c_int, order='C' )

    if dtype == numpy.float64:
        ppath  = <int*>    numpy.PyArray_DATA(path)
        palpha = <double*> numpy.PyArray_DATA(alpha)
        pA     = <double*> numpy.PyArray_DATA(A)
        ppobs  = <double*> numpy.PyArray_DATA(pobs)
        # call
        _sample_path(ppath, palpha, pA, ppobs, N, T)
        return path
    else:
        raise TypeError
