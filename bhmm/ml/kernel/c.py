import bhmm.ml.lib.c as ext
import numpy as np

def forward(A, pobs, pi, dtype=np.float32):
    print "dtype = ",dtype
    if dtype == np.float32:
        return ext.forward32(A, pobs, pi)
    if dtype == np.float64:
        print "AA"
        return ext.test()
        #return ext.forward(A, pobs, pi)
    else:
        raise ValueError

def backward(A, pobs, dtype=np.float32):
    if dtype == np.float32:
        return ext.backward32(A, pobs)
    if dtype == np.float64:
        return ext.backward(A, pobs)
    else:
        raise ValueError

def state_probabilities(alpha, beta, dtype=np.float32):
    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1)[:,None]
    return gamma

def state_counts(gamma, T, dtype=np.float32):
    return np.sum(gamma[0:T], axis=0)

def transition_probabilities(alpha, beta, A, B, ob, dtype=np.float32):
    if dtype == np.float32:
        return ext.transition_probabilities32(alpha, beta, A, B, ob)
    if dtype == np.float64:
        return ext.transition_probabilities(alpha, beta, A, B, ob)
    else:
        raise ValueError


# # TODO: I don't think it's worthing having this function in C. Test replacing it by a simple numpy multiplication.
# def state_probabilities(alpha, beta, dtype=np.float32):
#     if dtype == np.float32:
#         return ext.state_probabilities32(alpha, beta)
#     if dtype == np.float64:
#         return ext.state_probabilities(alpha, beta)
#     else:
#         raise ValueError
#
# # TODO: I don't think it's worthing having this function in C. Test replacing it by a simple numpy summation.
# def state_counts(gamma, T, dtype=np.float32):
#     if dtype == np.float32:
#         return ext.state_counts32(gamma, T)
#     if dtype == np.float64:
#         return ext.state_counts(gamma, T)
#     else:
#         raise ValueError
#
#
# def transition_probabilities(alpha, beta, A, B, ob, dtype=np.float32):
#     if dtype == np.float32:
#         return ext.transition_probabilities32(alpha, beta, A, B, ob)
#     if dtype == np.float64:
#         return ext.transition_probabilities(alpha, beta, A, B, ob)
#     else:
#         raise ValueError

def transition_counts(alpha, beta, A, B, ob, dtype=np.float32):
    if dtype == np.float32:
        return ext.transition_counts32(alpha, beta, A, B, ob)
    if dtype == np.float64:
        return ext.transition_counts(alpha, beta, A, B, ob)
    else:
        raise ValueError

def viterbi(A, pobs, pi):
    """ Generate an observation sequence of length T from the model A, B, pi.

    Parameters
    ----------
    A : numpy.array shape (N,N)
        transition matrix of the model
    pobs : numpy.array of floating numbers and shape (T,N)
        symbol probability matrix for each time and hidden state
    pi : numpy.array shape (N)
        starting probability vector of the model

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    T,N = pobs.shape[0], pobs.shape[1]
    # temporary viterbi state
    v = np.zeros((N))
    psi = np.zeros((T,N), dtype = int)
    # initialize
    v      = pi * pobs[0,:]
    # rescale
    v     /= v.sum()
    psi[0] = 0.0
    # iterate
    for t in range(1,T):
        vA = np.dot(np.diag(v), A)
        # propagate v
        v  = pobs[t,:] * np.max(vA, axis=0)
        # rescale
        v     /= v.sum()
        psi[t] = np.argmax(vA, axis=0)
    # iterate
    q = np.zeros((T), dtype = int)
    q[T-1] = np.argmax(v)
    for t in range(T-2, -1, -1):
        q[t] = psi[t+1,q[t+1]]
    # done
    return q