__author__ = 'noe'

import numpy as np
from scipy import linalg

def eigenvalues(A, n):
    """
    Helper function for computing the eigenvalues of A in a sorted way. Should be replaced by EMMA function

    :param A:
    :param n:
    :return:
    """
    v=linalg.eigvals(A).real
    idx=(-v).argsort()[:n]
    return v[idx]

def stationary_distribution(P):
    """
    Computes the stationary distribution of P. Should be replaced by EMMA function

    :param P:
    :return:
    """
    (v,w) = linalg.eig(P.T)
    mu = w[:,np.argmax(v)]
    return mu / np.sum(mu)