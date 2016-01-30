import msmtools
__author__ = 'noe'

import warnings

import numpy as np

from bhmm.hmm.generic_hmm import HMM
from bhmm.output_models.discrete import DiscreteOutputModel
from bhmm.util.logger import logger


def coarse_grain_transition_matrix(P, M, eps=0):
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
    P_coarse = np.maximum(P_coarse, eps)
    # and renormalize
    P_coarse /= P_coarse.sum(axis=1)[:, None]

    return P_coarse


def estimate_initial_coarse_graining(C, Prev, nstates, eps=None):
    """ Initial HMM based from MSM

    Parameters
    ----------
    C : ndarray(n, n)
        Count matrix used to obtain Prev
    Prev : ndarray(n, n)
        Reversible transition matrix. Used to computed metastable sets with PCCA.
    nstates : int
        Number of coarse states.
    eps : float or None
        Minimum output probability. Default: 0.01 / n

    Returns
    -------
    M : ndarray(n, m)
        membership matrix
    B : ndarray(m, n)
        output probabilities

    """
    # input
    n = Prev.shape[0]
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / n

    # check if we have enough MSM states to support the requested number of metastable states
    if nstates > n:
        raise NotImplementedError('Trying to initialize ' + str(nstates) + '-state HMM from smaller '
                                  + str(n) + '-state MSM.')

    # pcca
    from msmtools.analysis.dense.pcca import PCCA
    pcca_obj = PCCA(Prev, nstates)

    # memberships and output probabilities
    M = pcca_obj.memberships

    # full state space output matrix
    B = eps * np.ones((nstates, n), dtype=np.float64)
    # fill PCCA distributions if they exceed eps_B
    B = np.maximum(B, pcca_obj.output_probabilities)
    # renormalize B to make it row-stochastic
    B /= B.sum(axis=1)[:, None]

    return M, B


def estimate_initial_hmm(C, nstates, reversible=True, P=None, active=None,
                         eps_A=None, eps_B=None, separate=None):
    """Generate an initial HMM with discrete output densities

    Initializes HMM as described in [1]_. First estimates a Markov state model
    on the given observations, then uses PCCA+ to coarse-grain the transition
    matrix [2]_ which initializes the HMM transition matrix. The HMM output
    probabilities are given by Bayesian inversion from the PCCA+ memberships [1]_.

    The regularization parameters eps_A and eps_B are used
    to guarantee that the hidden transition matrix and output probability matrix
    have no zeros. HMM estimation algorithms such as the EM algorithm and the
    Bayesian sampling algorithm cannot recover from zero entries, i.e. once they
    are zero, they will stay zero.

    Parameters
    ----------
    C : ndarray(n, n)
        Transition count matrix
    nstates : int
        The number of hidden states.
    reversible : bool
        Estimate reversible HMM transition matrix.
    P : ndarray(n, n)
        Transition matrix estimated from C (with option reversible). Use this
        option if P has already been estimated to avoid estimating it twice.
    active : ndarray(N, dtype=bool)
        Boolean array marking which states of the full state space are used by C and P
    eps_A : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_B : float or None
        Minimum output probability. Default: 0.01 / nfull
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.

    Raises
    ------
    NotImplementedError
        If the number of hidden states exceeds the number of observed states.

    Examples
    --------
    Generate initial model for a discrete output model.

    >>> import numpy as np
    >>> C = np.array([[0.5, 0.5, 0.0], [0.4, 0.5, 0.1], [0.0, 0.1, 0.9]])
    >>> initial_model = estimate_initial_hmm(C, 2)

    References
    ----------
    .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
        Markov models for calculating kinetics and  metastable states of
        complex molecules. J. Chem. Phys. 139, 184114 (2013)
    .. [2] S. Kube and M. Weber: A coarse graining method for the identification
        of transition rates between molecular conformations.
        J. Chem. Phys. 126, 024103 (2007)

    """
    # local imports
    from bhmm.estimators import _tmatrix_disconnected
    import sys

    # MICROSTATE COUNT MATRIX
    C_full = C
    nfull = C_full.shape[0]

    # INPUTS
    if eps_A is None:  # default transition probability, in order to avoid zero columns
        eps_A = 0.01 / nstates
    if eps_B is None:  # default output probability, in order to avoid zero columns
        eps_B = 0.01 / nfull

    # truncate to states with at least one observed incoming or outgoing count.
    nonempty = _tmatrix_disconnected.nonempty_set(C_full)
    C_nonempty = C_full[np.ix_(nonempty, nonempty)]

    # when using separate states, only keep the nonempty ones (the others don't matter)
    if separate is None:
        nonseparate = nonempty.copy()
        nmeta = nstates
    else:
        separate = np.array(list(set(separate).intersection(set(nonempty))))
        nonseparate = np.array(list(set(nonempty) - set(separate)))
        nmeta = nstates - 1

    # check if we can proceed
    if nonseparate.size < nmeta:
        raise NotImplementedError('Trying to initialize ' + str(nmeta) + '-state HMM from smaller '
                                  + str(nonseparate.size) + '-state MSM.')

    # MICROSTATE TRANSITION MATRIX (MSM). This matrix may be disconnected and have transient states
    if P is not None:
        P_msm = P[nonempty, :][:, nonempty]  # if we already have P, just remove empty states
    else:
        P_msm = _tmatrix_disconnected.estimate_P(C_nonempty, reversible=reversible)
    pi_msm = _tmatrix_disconnected.stationary_distribution(C_nonempty, P_msm)
    pi_full = np.zeros(nfull)
    pi_full[nonempty] = pi_msm

    # NONSEPARATE TRANSITION MATRIX FOR PCCA+
    C_nonseparate = C_full[np.ix_(nonseparate, nonseparate)]
    if reversible and nonseparate.size == nonempty.size:  # in this case we already have a reversible estimate that is large enough
        P_for_pcca = P_msm
    else:  # not yet reversible. re-estimate
        P_for_pcca = _tmatrix_disconnected.estimate_P(C_nonseparate, reversible=True)

    # COARSE-GRAINING
    if nonseparate.size > nmeta:
        M_ns, B_ns = estimate_initial_coarse_graining(C_nonseparate, P_for_pcca, nmeta, eps=eps_B)
    else:  # equal size
        M_ns = np.eye(nmeta)
        B_ns = np.eye(nmeta)

    # MEMBERSHIP
    M = np.zeros((nfull, nstates))
    M[nonseparate, :nmeta] = M_ns
    if separate is not None:  # add separate states
        M[separate, -1] = 1

    # COARSE-GRAINED TRANSITION MATRIX
    P_coarse = coarse_grain_transition_matrix(P_msm, M[nonempty], eps=eps_A)
    if reversible:
        P_coarse = _tmatrix_disconnected.enforce_reversible_on_closed(P_coarse)
    C_coarse = M[nonempty].T.dot(C_nonempty).dot(M[nonempty])
    pi_coarse = _tmatrix_disconnected.stationary_distribution(C_coarse, P_coarse)  # need C_coarse in case if A is disconnected

    # COARSE-GRAINED OUTPUT DISTRIBUTION
    B = np.zeros((nstates, nfull))
    B[:nmeta, nonseparate] = B_ns
    if separate is not None:  # add separate states
        B[-1, separate] = pi_full[separate]
    B /= B.sum(axis=1)[:, None]  # normalize rows

    # print 'cg pi: ', pi_coarse
    # print 'cg A:\n ', P_coarse
    # print 'cg B:\n ', B

    logger().info('Initial model: ')
    logger().info('initial distribution = \n'+str(pi_coarse))
    logger().info('transition matrix = \n'+str(P_coarse))
    logger().info('output matrix = \n'+str(B.T))

    # initialize HMM
    output_model = DiscreteOutputModel(B)
    model = HMM(pi_coarse, P_coarse, output_model)
    return model


