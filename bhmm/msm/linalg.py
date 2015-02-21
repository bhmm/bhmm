__author__ = 'noe'

import numpy as np
from scipy import linalg

def eigenvalues(A, n):
    """
    Return the eigenvalues of A in order from largest to smallest.

    Parameters
    ----------
    A : numpy.array with shape (nstates,nstates)
        The matrix for which eigenvalues are to be computed.
    n : int
        The number of largest eigenvalues to return.

    Examples
    --------

    Return all sorted eigenvalues.

    >>> from bhmm import testsystems
    >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
    >>> ew_sorted = eigenvalues(Tij, 3)

    Return largest eigenvalue.

    >>> ew_sorted = eigenvalues(Tij, 1)

    TODO
    ----
    Replace this with a call to the EMMA method once we use EMMA as a dependency.

    """
    v=linalg.eigvals(A).real
    idx=(-v).argsort()[:n]
    return v[idx]

def stationary_distribution(P):
    """
    Computes the stationary distribution of a row-stochastic transition matrixn

    Parameters
    ----------
    P : numpy.array with shape (nstates,nstates)
        The row-stochastic transition matrix.

    Examples
    --------

    Compute stationary probabilities for a given transition matrix.

    >>> from bhmm import testsystems
    >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
    >>> Pi = stationary_distribution(Tij)

    TODO
    ----
    Replace this with a call to the EMMA method once we use EMMA as a dependency.

    """
    (v,w) = linalg.eig(P.T)
    mu = w[:,np.argmax(v)]
    return mu / np.sum(mu)
