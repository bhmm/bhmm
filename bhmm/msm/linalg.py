__author__ = 'noe'

import numpy as np
from scipy import linalg

__author__ = "Benjamin-Trendelkamp Schroer, Martin Scherer, Fabian Paul, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["Benjamin-Trendelkamp Schroer", "Martin Scherer", "Fabian Paul", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "Martin Scherer"
__email__="martin DOT scherer AT fu-berlin DOT de"


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

    >>> from bhmm.util import testsystems
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

    >>> from bhmm.util import testsystems
    >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
    >>> Pi = stationary_distribution(Tij)

    TODO
    ----
    Replace this with a call to the EMMA method once we use EMMA as a dependency.

    """
    if not is_connected(P):
        raise ValueError('Cannot calculate stationary distribution. Input matrix is not connected:\n ',str(P))
    (v,w) = linalg.eig(P.T)
    mu = w[:,np.argmax(v)]
    mu0 = mu / np.sum(mu)
    np.real(mu0)
    return mu0


def transition_matrix_MLE_nonreversible(C):
    r"""
    Estimates a nonreversible transition matrix from count matrix C
    (EMMA function)

    T_ij = c_ij / c_i where c_i = sum_j c_ij

    Parameters
    ----------
    C: ndarray, shape (n,n)
        count matrix

    Returns
    -------
    T: Estimated transition matrix

    """
    # multiply by 1.0 to make sure we're not doing integer division
    rowsums = 1.0*np.sum(C,axis=1)
    if np.min(rowsums) <= 0:
        raise ValueError("Transition matrix has row sum of "+str(np.min(rowsums))+". Must have strictly positive row sums.")
    return C / rowsums[:,np.newaxis]


def __initX(C):
    """
    Computes an initial guess for a reversible correlation matrix
    (EMMA function)

    """
    T = transition_matrix_MLE_nonreversible(C)
    mu = stationary_distribution(T)
    Corr = np.dot(np.diag(mu), T)
    return 0.5 * (Corr + Corr.T)


def __relative_error(x, y, norm=None):
    """
    computes the norm of the vector with elementwise relative errors
    between x and y, defined by (x_i - y_i) / (x_i + y_i)
    (EMMA function)

    x : ndarray (n)
        vector 1
    y : ndarray (n)
        vector 2
    norm : vector norm to be used. By default the Euclidean norm is used.
        This value is passed as 'ord' to numpy.linalg.norm()

    """
    d = (x - y)
    s = (x + y)
    # to avoid dividing by zero, always set to 0
    nz = np.nonzero(d)
    # relative error vector
    erel = d[nz] / s[nz]
    # return euclidean norm
    return np.linalg.norm(erel, ord=norm)


def is_connected(C, directed=True):
    r"""Return true, if the input count matrix is completely connected.
    Effectively checking if the number of connected components equals one.
    (EMMA function)

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        Count matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.

    Returns
    -------
    connected : boolean, returning true only if C is connected.


    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.sputils import isdense
    import scipy.sparse.csgraph as csgraph
    if isdense(C):
        C = csr_matrix(C)
    nc=csgraph.connected_components(C, directed=directed, connection='strong', return_labels=False)
    return nc == 1


def log_likelihood(C, T):
    r"""Log-likelihood of the count matrix given a transition matrix.
    (EMMA function)

    Parameters
    ----------
    C : (M, M) ndarray or scipy.sparse matrix
        Count matrix
    T : (M, M) ndarray orscipy.sparse matrix
        Transition matrix

    Returns
    -------
    logL : float
        Log-likelihood of the count matrix

    Notes
    -----

    The likelihood of a set of observed transition counts
    :math:`C=(c_{ij})` for a given matrix of transition counts
    :math:`T=(t_{ij})` is given by

    .. math:: L(C|P)=\prod_{i=1}^{M} \left( \prod_{j=1}^{M} p_{ij}^{c_{ij}} \right)

    The log-likelihood is given by

    .. math:: l(C|P)=\sum_{i,j=1}^{M}c_{ij} \log p_{ij}.

    The likelihood describes the probability of making an observation
    :math:`C` for a given model :math:`P`.

    Examples
    --------

    >>> T=np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])

    >>> C=np.array([[58, 7, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL=log_likelihood(C, T)
    >>> logL
    -38.280803472508182

    >>> C=np.array([[58, 20, 0], [6, 0, 4], [0, 3, 21]])
    >>> logL=log_likelihood(C, T)
    >>> logL
    -68.214409681430766

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105

    """
    # use the dense likelihood calculator for all other cases
    # if a mix of dense/sparse C/T matrices is used, then both
    # will be converted to ndarrays.
    if (not isinstance(C, np.ndarray)):
        C = np.array(C)
    if (not isinstance(T, np.ndarray)):
        T = np.array(T)
    # computation is still efficient, because we only use terms
    # for nonzero elements of T
    nz = np.nonzero(T)
    return np.dot(C[nz], np.log(T[nz]))


def transition_matrix_MLE_reversible(C, Xinit = None, maxiter = 1000000, maxerr = 1e-8, return_statdist = False, return_conv = False):
    """
    iterative method for estimating a maximum likelihood reversible transition matrix

    The iteration equation implemented here is:
        t_ij = (c_ij + c_ji) / ((c_i / x_i) + (c_j / x_j))
    Please note that there is a better (=faster) iteration that has been described in
    Prinz et al, J. Chem. Phys. 134, p. 174105 (2011). We should implement that too.
    (EMMA function)

    Parameters
    ----------
    C : ndarray (n,n)
        count matrix. If a non-connected count matrix is used, the method returns in error
    Xinit = None : ndarray (n,n)
        initial value for the matrix of absolute transition probabilities. Unless set otherwise,
        will use X = diag(pi) T, where T is a nonreversible transition matrix estimated from C,
        i.e. T_ij = c_ij / sum_k c_ik, and pi is its stationary distribution.
    maxerr = 1000000 : int
        maximum number of iterations before the method exits
    maxiter = 1e-8 : float
        convergence tolerance. This specifies the maximum change of the Euclidean norm of relative
        stationary probabilities (x_i = sum_k x_ik). The relative stationary probability changes
        e_i = (x_i^(1) - x_i^(2))/(x_i^(1) + x_i^(2)) are used in order to track changes in small
        probabilities. The Euclidean norm of the change vector, |e_i|_2, is compared to convtol.
    return_statdist = False : Boolean
        If set to true, the stationary distribution is also returned
    return_conv = False : Boolean
        If set to true, the likelihood history and the pi_change history is returned.

    Returns
    -------
    T or (T,pi) or (T,lhist,pi_changes) or (T,pi,lhist,pi_changes)
    T : ndarray (n,n)
        transition matrix. This is the only return for return_statdist = False, return_conv = False
    (pi) : ndarray (n)
        stationary distribution. Only returned if return_statdist = True
    (lhist) : ndarray (k)
        likelihood history. Has the length of the number of iterations needed.
        Only returned if return_conv = True
    (pi_changes) : ndarray (k)
        history of likelihood history. Has the length of the number of iterations needed.
        Only returned if return_conv = True
    """
    # check input
    if (not is_connected(C)):
        ValueError('Count matrix is not fully connected. '+
                   'Need fully connected count matrix for '+
                   'reversible transition matrix estimation.')
    converged = False
    n = np.shape(C)[0]
    # initialization
    C2 = C + C.T # reversibly counted matrix
    nz = np.nonzero(C2)
    csum = np.sum(C, axis=1) # row sums C
    X = Xinit
    if (X is None):
        X = __initX(C) # initial X
    xsum = np.sum(X, axis=1) # row sums x
    D = np.zeros((n,n)) # helper matrix
    T = np.zeros((n,n)) # transition matrix
    # if convergence history requested, initialize variables
    if (return_conv):
        diffs = np.zeros(maxiter)
        # likelihood
        lhist = np.zeros(maxiter)
        T = X / xsum[:,np.newaxis]
        lhist[0] = log_likelihood(C, T)
    # iteration
    i = 1
    while (i < maxiter-1) and (not converged):
        # c_i / x_i
        c_over_x = csum / xsum
        # d_ij = (c_i/x_i) + (c_j/x_j)
        D[:] = c_over_x[:,np.newaxis]
        D += c_over_x
        # update estimate
        X[nz] = C2[nz] / D[nz]
        X[nz] /= np.sum(X[nz])         # renormalize
        xsumnew = np.sum(X, axis=1)
        # compute difference in pi
        diff = __relative_error(xsum, xsumnew)
        # update pi
        xsum = xsumnew
        # any convergence history wanted?
        if (return_conv):
            # update T and likelihood
            T = X / xsum[:,np.newaxis]
            lhist[i] = log_likelihood(C, T)
            diffs[i] = diff
        # converged?
        converged = (diff < maxerr)
        i += 1
    # finalize and return
    T = X / xsum[:,np.newaxis]
    if (return_statdist and return_conv):
        return (T, xsum, lhist[0:i], diffs[0:i])
    if (return_statdist):
        return (T, xsum)
    if (return_conv):
        return (T, lhist[0:i], diffs[0:i])
    return T # else just return T

#
# def main():
#     C = np.array([[5,2,1],
#                   [1,2,3],
#                   [5,6,10]])
#     (P,ll,dd) = transition_matrix_MLE_reversible(C, return_conv=True)
#     print P
#     print ll
#
#     pi = stationary_distribution(P)
#
#     Corr = np.dot(np.diag(pi),P)
#     print Corr
#
# if __name__ == '__main__':
#     main()
