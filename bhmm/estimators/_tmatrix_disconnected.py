__author__ = 'noe'

import numpy as np


def connected_sets(C, mincount_connectivity=0, strong=True):
    """ Computes the connected sets of C.

    C : count matrix
    mincount_connectivity : float
        Minimum count which counts as a connection.
    strong : boolean
        True: Seek strongly connected sets. False: Seek weakly connected sets.

    """
    import msmtools.estimation as msmest
    Cconn = C.copy()
    Cconn[np.where(C < mincount_connectivity)] = 0
    # treat each connected set separately
    S = msmest.connected_sets(Cconn, directed=strong)
    return S


def estimate_P(C, reversible=True, fixed_statdist=None, maxiter=1000000, maxerr=1e-8, mincount_connectivity=0):
    """ Estimates full transition matrix for general connectivity structure

    Parameters
    ----------
    C : ndarray
        count matrix
    reversible : bool
        estimate reversible?
    fixed_statdist : ndarray or None
        estimate with given stationary distribution
    maxiter : int
        Maximum number of reversible iterations.
    maxerr : float
        Stopping criterion for reversible iteration: Will stop when infinity
        norm  of difference vector of two subsequent equilibrium distributions
        is below maxerr.
    mincount_connectivity : float
        Minimum count which counts as a connection.

    """
    import msmtools.estimation as msmest
    n = np.shape(C)[0]
    # output matrix. Set initially to Identity matrix in order to handle empty states
    P = np.eye(n, dtype=np.float64)
    # decide if we need to proceed by weakly or strongly connected sets
    if reversible and fixed_statdist is None:  # reversible to unknown eq. dist. - use strongly connected sets.
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=True)
        for s in S:
            mask = np.zeros(n, dtype=bool)
            mask[s] = True
            if C[np.ix_(mask, ~mask)].sum() > 0:  # outgoing transitions - use partial rev algo.
                transition_matrix_partial_rev(C, P, mask, maxiter=maxiter, maxerr=maxerr)
            else:  # closed set - use standard estimator
                I = np.ix_(mask, mask)
                if s.size > 1:  # leave diagonal 1 if single closed state.
                    P[I] = msmest.transition_matrix(C[I], reversible=True, warn_not_converged=False,
                                                    maxiter=maxiter, maxerr=maxerr)
    else:  # nonreversible or given equilibrium distribution - weakly connected sets
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
        for s in S:
            I = np.ix_(s, s)
            if not reversible:
                Csub = C[I]
                # any zero rows? must set Cii = 1 to avoid dividing by zero
                zero_rows = np.where(Csub.sum(axis=1) == 0)[0]
                Csub[zero_rows, zero_rows] = 1.0
                P[I] = msmest.transition_matrix(Csub, reversible=False)
            elif reversible and fixed_statdist is not None:
                P[I] = msmest.transition_matrix(C[I], reversible=True, fixed_statdist=fixed_statdist,
                                                maxiter=maxiter, maxerr=maxerr)
            else:  # unknown case
                raise NotImplementedError('Transition estimation for the case reversible=' + str(reversible) +
                                          ' fixed_statdist=' + str(fixed_statdist is not None) + ' not implemented.')
    # done
    return P


def transition_matrix_partial_rev(C, P, S, maxiter=1000000, maxerr=1e-8):
    """Maximum likelihood estimation of transition matrix which is reversible on parts

    Partially-reversible estimation of transition matrix. Maximizes the likelihood:

    .. math:
        P_S &=& arg max prod_{S, :} (p_ij)^c_ij \\
        \Pi_S P_{S,S} &=& \Pi_S P_{S,S}

    where the product runs over all elements of the rows S, and detailed balance only
    acts on the block with rows and columns S. :math:`\Pi_S` is the diagonal matrix of
    equilibrium probabilities restricted to set S.

    Note that this formulation

    Parameters
    ----------
    C : ndarray
        full count matrix
    P : ndarray
        full transition matrix to write to. Will overwrite P[S]
    S : ndarray, bool
        boolean selection of reversible set with outgoing transitions
    maxerr : float
        maximum difference in matrix sums between iterations (infinity norm) in order to stop.

    """
    # test input
    assert np.array_equal(C.shape, P.shape)
    # constants
    A = C[S][:, S]
    B = C[S][:, ~S]
    ATA = A + A.T
    countsums = C[S].sum(axis=1)
    # initialize
    X = 0.5 * ATA
    Y = C[S][:, ~S]
    # normalize X, Y
    totalsum = X.sum() + Y.sum()
    X /= totalsum
    Y /= totalsum
    # rowsums
    rowsums = X.sum(axis=1) + Y.sum(axis=1)
    err = 1.0
    it = 0
    while err > maxerr and it < maxiter:
        # update
        d = countsums / rowsums
        X = ATA / (d[:, None] + d)
        Y = B / d[:, None]
        # normalize X, Y
        totalsum = X.sum() + Y.sum()
        X /= totalsum
        Y /= totalsum
        # update sums
        rowsums_new = X.sum(axis=1) + Y.sum(axis=1)
        # compute error
        err = np.max(np.abs(rowsums_new - rowsums))
        # update
        rowsums = rowsums_new
        it += 1
    # write to P
    P[np.ix_(S, S)] = X
    P[np.ix_(S, ~S)] = Y
    P[S] /= P[S].sum(axis=1)[:, None]


def enforce_reversible(P):
    """ Enforces transition matrix P to be reversible. """
    # compute stationary probability
    from msmtools.analysis import stationary_distribution
    pi = stationary_distribution(P)
    # symmetrize
    X = np.dot(np.diag(pi), P)
    X = 0.5 * (X + X.T)
    # normalize
    Prev = X / X.sum(axis=1)[:, None]
    return Prev


def stationary_distribution(C, P, mincount_connectivity=0):
    """ Simple estimator for stationary distribution for multiple strongly connected sets """
    # can be replaced by msmtools.analysis.stationary_distribution in next msmtools release
    from msmtools.analysis.dense.stationary_vector import stationary_distribution as msmstatdist
    # disconnected sets
    n = np.shape(C)[0]
    ctot = np.sum(C)
    pi = np.zeros(n)
    # treat each weakly connected set separately
    sets = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
    for s in sets:
        # compute weight
        w = np.sum(C[s, :]) / ctot
        pi[s] = w * msmstatdist(P[s, :][:, s])
    # reinforce normalization
    pi /= np.sum(pi)
    return pi


def rdl_decomposition(P, reversible=True):
    # TODO: this treatment is probably not meaningful for weakly connected matrices.
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    # output matrices
    n = np.shape(P)[0]
    if reversible:
        dtype = np.float64
    else:
        dtype = complex
    R = np.zeros((n, n), dtype=dtype)
    D = np.zeros((n, n), dtype=dtype)
    L = np.zeros((n, n), dtype=dtype)
    # treat each strongly connected set separately
    S = msmest.connected_sets(P)
    for s in S:
        I = np.ix_(s, s)
        if len(s) > 1:
            if reversible:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='reversible')
                # everything must be real-valued - this should rather be handled by msmtools
                R[I] = r.real
                D[I] = d.real
                L[I] = l.real
            else:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='standard')
                # write to full
                R[I] = r
                D[I] = d
                L[I] = l
        else:  # just one element. Write 1's
            R[I] = 1
            D[I] = 1
            L[I] = 1
    # done
    return R, D, L
