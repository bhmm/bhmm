import msmtools
__author__ = 'noe'

import warnings

import numpy as np

from bhmm.hmm.generic_hmm import HMM
from bhmm.output_models.discrete import DiscreteOutputModel
from bhmm.util.logger import logger


# TODO: check availability of lag_observations as in PyEMMA / MaximumLikelihoodHMSM

def coarse_grain_transition_matrix(P, M, epsilon=0):
    """ Coarse grain transition matrix P using memberships M

    Computes

    .. math:
        Pc = (M' M)^-1 M' P M

    Parameters
    ----------
    P : ndarray(n, n)
        microstate transition matrix
    M : ndarray(n, m)
        membership matrix. Membership to macrostate m for each microstate.
    epsilon : float
        minimum value of the resulting transition matrix. The coarse-graining
        equation can lead to negative elements and thus epsilon should be set to
        at least 0. Positive settings of epsilon are similar to a prior and
        enforce minimum positive values for all transition probabilities.

    Returns
    -------
    Pc : ndarray(m, m)
        coarse-grained transition matrix.

    """
    # coarse-grain matrix: Pc = (M' M)^-1 M' P M
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # this coarse-graining can lead to negative elements. Setting them to epsilon here.
    P_coarse = np.maximum(P_coarse, epsilon)
    # and renormalize
    P_coarse /= P_coarse.sum(axis=1)[:, None]

    return P_coarse


# TODO: does not implement fixed stationary distribution (on macrostates) yet.
def msm_to_hmm(Prev, nstates, P=None, indexes=None, nfull=None, bmin=None):
    """ Initial HMM based from MSM

    Parameters
    ----------
    Prev : ndarray(n, n)
        Reversible transition matrix. Used to computed metastable sets with PCCA.
    nstates : int
        Number of coarse states.
    P : ndarray(n, n)
        Transition matrix to be coarse-grained for the initial HMM. If P=None,
        will use P=Prev.
    indexes : ndarray(n) or None
        Mapping from rows of Prev and P to state indexes of the full state space.
        This should be used, if Prev and P only live on a subset of states.
        None means, that the indexes of Prev/P rows are identical to the full
        state space.
    nfull : int or None
        Number of states in full state space. If None, nfull = Pref.shape[0].
    bmin : float or None
        Minimum output probability. Default: 0.01 / nfull

    """
    # input
    if P is None:
        P = Prev
    n = Prev.shape[0]
    if nfull is None:
        nfull = n
    else:
        assert indexes is not None, 'If nfull is specified, must specify indexes too.'
    if indexes is None:
        indexes = np.arange(nfull)
    if bmin is None:  # default output probability, in order to avoid zero columns
        bmin = 0.01 / nfull

    print 'Coarse-graining: \n', Prev

    # coarse-grain
    if nstates > n:
        raise NotImplementedError('Trying to initialize '+str(nstates)+'-state HMM from smaller '+str(n)+'-state MSM.')
    elif nstates == n:
        A = P
        B = np.eye(n)
    else:
        # pcca
        from msmtools.analysis.dense.pcca import PCCA
        pcca_obj = PCCA(Prev, nstates)
        # Use PCCA distributions, but avoid 100% assignment to any state (prevents convergence)
        B_conn = np.maximum(pcca_obj.output_probabilities, bmin)
        # full state space output matrix
        B = bmin * np.ones((nstates, nfull), dtype=np.float64)
        # expand B_conn to full state space
        B[:, indexes] = B_conn[:, :]
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:, None]
        # coarse-grained transition matrix
        A = coarse_grain_transition_matrix(P, pcca_obj.memberships)

        print 'Memberships = \n', pcca_obj.memberships

    return A, B


def estimate_initial_model(observations, nstates, reversible=True, prior_neighbor=0.0, prior_diag=0.0):
    """Generate an initial model with discrete output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=int)
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    revprior : reversible prior counts using in MSM transition matrix estimation
        before it is coarse-grained. Defined by b_ij = revprior for all i,j with
        c_ij+c_ji > 0.

    Examples
    --------

    Generate initial model for a discrete output model.

    >>> from bhmm import testsystems
    >>> [model, observations, states] = testsystems.generate_synthetic_observations(output_model_type='discrete')
    >>> initial_model = estimate_initial_model(observations, model.nstates)

    """
    # full count matrix at lag 1 (do we need a general lag here?)
    C_full = msmtools.estimation.count_matrix(observations, 1)
    print 'Cobs\n', C_full
    if prior_neighbor > 0:
        # prior matrix
        B = msmtools.estimation.prior_neighbor(C_full, alpha=prior_neighbor)
        print 'B_neighbor\n', B
        # posterior count matrix
        C_full += B
        print 'Cpost\n', C_full
    # make dense, because we don't have sparse implementations of the next steps yet
    C_full = C_full.toarray()
    if prior_diag > 0:  # posterior count matrix
        C_full += prior_diag * np.eye(C_full.shape[0])
        print 'Cpost+diag\n', C_full
    # truncate to states with at least one observed incoming or outgoing count.
    indexes = np.where(C_full.sum(axis=0) + C_full.sum(axis=1) > 0)[0]
    C = C_full[np.ix_(indexes, indexes)]

    print 'C\n', C

    # estimate reversible transition matrix-needed for PCCA
    # Note that states with zero rows will get a diagonal 1 in the resulting P.
    from bhmm.estimators._tmatrix_disconnected import estimate_P, enforce_reversible
    Prev = estimate_P(C, reversible=True)
    P = None
    # if needed, also estimate nonreversible transition matrix
    if not reversible:
        P = estimate_P(C, reversible=False)

    print 'P_micro_rev\n', Prev
    print 'P_micro\n', P

    # initialize HMM
    A, B = msm_to_hmm(Prev, nstates, P=P, indexes=indexes, nfull=C_full.shape[0])
    if reversible:
        A = enforce_reversible(A)

    print 'A\n', A
    print 'B\n', B

    logger().info('Initial model: ')
    logger().info('transition matrix = \n'+str(A))
    logger().info('output matrix = \n'+str(B.T))

    # initialize HMM
    output_model = DiscreteOutputModel(B)
    model = HMM(A, output_model, reversible=reversible)
    return model


