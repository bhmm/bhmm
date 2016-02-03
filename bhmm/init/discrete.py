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


def estimate_initial_hmm(C_full, nstates, reversible=True, active_set=None, P=None,
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
    C_full : ndarray(N, N)
        Transition count matrix on the full observable state space
    nstates : int
        The number of hidden states.
    reversible : bool
        Estimate reversible HMM transition matrix.
    active_set : ndarray(n, dtype=int) or None
        Index area. Will estimate kinetics only on the given subset of C
    P : ndarray(n, n)
        Transition matrix estimated from C (with option reversible). Use this
        option if P has already been estimated to avoid estimating it twice.
    eps_A : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_B : float or None
        Minimum output probability. Default: 0.01 / nfull
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.

    Raises
    ------
    ValueError
        If the given active set is illegal.
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

    # MICROSTATE COUNT MATRIX
    nfull = C_full.shape[0]

    # INPUTS
    if eps_A is None:  # default transition probability, in order to avoid zero columns
        eps_A = 0.01 / nstates
    if eps_B is None:  # default output probability, in order to avoid zero columns
        eps_B = 0.01 / nfull
    # Manage sets
    symsum = C_full.sum(axis=0) + C_full.sum(axis=1)
    nonempty = np.where(symsum > 0)[0]
    if active_set is None:
        active_set = nonempty
    else:
        if np.any(symsum[active_set] == 0):
            raise ValueError('Given active set has empty states')  # don't tolerate empty states
    if P is not None:
        if np.shape(P)[0] != active_set.size:  # needs to fit to active
            raise ValueError('Given initial transition matrix P has shape ' + str(np.shape(P))
                             + 'while active set has size ' + str(active_set.size))
    # when using separate states, only keep the nonempty ones (the others don't matter)
    if separate is None:
        active_nonseparate = active_set.copy()
        nmeta = nstates
    else:
        if np.max(separate) >= nfull:
            raise ValueError('Separate set has indexes that do not exist in full state space: '
                             + str(np.max(separate)))
        active_nonseparate = np.array(list(set(active_set) - set(separate)))
        nmeta = nstates - 1
    # check if we can proceed
    if active_nonseparate.size < nmeta:
        raise NotImplementedError('Trying to initialize ' + str(nmeta) + '-state HMM from smaller '
                                  + str(active_nonseparate.size) + '-state MSM.')

    # MICROSTATE TRANSITION MATRIX (MSM).
    C_active = C_full[np.ix_(active_set, active_set)]
    if P is None:  # This matrix may be disconnected and have transient states
        P_active = _tmatrix_disconnected.estimate_P(C_active, reversible=reversible)
    else:
        P_active = P

    # MICROSTATE EQUILIBRIUM DISTRIBUTION
    pi_active = _tmatrix_disconnected.stationary_distribution(C_active, P_active)
    pi_full = np.zeros(nfull)
    pi_full[active_set] = pi_active

    # NONSEPARATE TRANSITION MATRIX FOR PCCA+
    C_active_nonseparate = C_full[np.ix_(active_nonseparate, active_nonseparate)]
    if reversible and separate is None:  # in this case we already have a reversible estimate with the right size
        P_active_nonseparate = P_active
    else:  # not yet reversible. re-estimate
        P_active_nonseparate = _tmatrix_disconnected.estimate_P(C_active_nonseparate, reversible=True)

    # COARSE-GRAINING WITH PCCA+
    if active_nonseparate.size > nmeta:
        from msmtools.analysis.dense.pcca import PCCA
        pcca_obj = PCCA(P_active_nonseparate, nmeta)
        M_active_nonseparate = pcca_obj.memberships  # memberships
        B_active_nonseparate = pcca_obj.output_probabilities  #  output probabilities
    else:  # equal size
        M_active_nonseparate = np.eye(nmeta)
        B_active_nonseparate = np.eye(nmeta)

    # ADD SEPARATE STATE IF NEEDED
    if separate is None:
        M_active = M_active_nonseparate
    else:
        M_full = np.zeros((nfull, nstates))
        M_full[active_nonseparate, :nmeta] = M_active_nonseparate
        M_full[separate, -1] = 1
        M_active = M_full[active_set]

    # COARSE-GRAINED TRANSITION MATRIX
    P_hmm = coarse_grain_transition_matrix(P_active, M_active, eps=eps_A)
    if reversible:
        P_hmm = _tmatrix_disconnected.enforce_reversible_on_closed(P_hmm)
    C_hmm = M_active.T.dot(C_active).dot(M_active)
    pi_hmm = _tmatrix_disconnected.stationary_distribution(C_hmm, P_hmm)  # need C_hmm in case if A is disconnected

    # COARSE-GRAINED OUTPUT DISTRIBUTION
    # TODO: add eps_B to nonempty nonactive
    B_hmm = np.zeros((nstates, nfull))
    B_hmm[:nmeta, active_nonseparate] = np.maximum(B_active_nonseparate, eps_B)  # never allow 0.0 or 1.0
    if separate is not None:  # add separate states
        B_hmm[-1, separate] = pi_full[separate]
    nonempty_nonactive = np.array(list(set(nonempty) - set(active_set)), dtype=int)
    B_hmm[:, nonempty_nonactive] = eps_B
    B_hmm /= B_hmm.sum(axis=1)[:, None]  # normalize rows

    # print 'cg pi: ', pi_hmm
    # print 'cg A:\n ', P_hmm
    # print 'cg B:\n ', B

    logger().info('Initial model: ')
    logger().info('initial distribution = \n'+str(pi_hmm))
    logger().info('transition matrix = \n'+str(P_hmm))
    logger().info('output matrix = \n'+str(B_hmm.T))

    # initialize HMM
    model = HMM(pi_hmm, P_hmm, DiscreteOutputModel(B_hmm))
    return model


