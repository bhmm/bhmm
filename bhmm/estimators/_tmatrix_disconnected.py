__author__ = 'noe'

import numpy as np


def estimate_P(C, reversible=True, fixed_statdist=None, mincount_connectivity=1e-2):
    """ Estimates transition matrix for general connectivity structure

    Parameters
    ----------
    C : ndarray
        count matrix
    reversible : bool
        estimate reversible?
    fixed_statdist : ndarray or None
        estimate with given stationary distribution
    mincount_connectivity
        minimum number of counts to consider a connection between two states. Counts
        lower than that will count zero in the connectivity check and may thus separate
        the resulting transition matrix.

    """
    import msmtools.estimation as msmest
    # output matrix. Initially eye
    n = np.shape(C)[0]
    P = np.eye(n, dtype=np.float64)
    # count matrix for connectivity check
    Cconn = C.copy()
    Cconn[np.where(C < mincount_connectivity)] = 0
    # treat each connected set separately
    S = msmest.connected_sets(Cconn)
    for s in S:
        if len(s) > 1:  # if there's only one state, there's nothing to estimate and we leave it with diagonal 1
            # compute transition sub-matrix on s
            Cs = C[s, :][:, s]
            Ps = msmest.transition_matrix(Cs, reversible=reversible, mu=fixed_statdist)
            # write back to matrix
            for i, I in enumerate(s):
                for j, J in enumerate(s):
                    P[I, J] = Ps[i, j]
            P[s, :][:, s] = Ps
    # done
    return P

def stationary_distribution(C, P):
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    # disconnected sets
    n = np.shape(C)[0]
    ctot = np.sum(C)
    pi = np.zeros(n)
    # treat each connected set separately
    S = msmest.connected_sets(C)
    for s in S:
        # compute weight
        w = np.sum(C[s, :]) / ctot
        pi[s] = w * msmana.stationary_distribution(P[s, :][:, s])
    # reinforce normalization
    pi /= np.sum(pi)
    return pi

def rdl_decomposition(P, reversible=True):
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    # output matrices
    n = np.shape(P)[0]
    R = np.zeros((n, n))
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    # treat each connected set separately
    S = msmest.connected_sets(P)
    for s in S:
        if len(s) > 1:
            if reversible:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='reversible')
                # everything must be real-valued - this should rather be handled by msmtools
                r = r.real
                d = d.real
                l = l.real
            else:
                r, d, l = msmana.rdl_decomposition(P[s, :][:, s], norm='standard')
            # write to full
            R[s, :][:, s] = r
            D[s, :][:, s] = d
            L[s, :][:, s] = l
        else:  # just one element. Write 1's
            R[s, :][:, s] = 1
            D[s, :][:, s] = 1
            L[s, :][:, s] = 1
    # done
    return R, D, L
