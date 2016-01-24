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


# TODO: does not implement fixed stationary distribution (on macrostates) yet.
def msm_to_hmm(C, Prev, nstates, P=None, indexes=None, nfull=None, eps_A=None, eps_B=None):
    """ Initial HMM based from MSM

    Parameters
    ----------
    C : ndarray(n, n)
        Count matrix used to obtain Prev
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
    eps_A : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_B : float or None
        Minimum output probability. Default: 0.01 / nfull

    """
    from bhmm.estimators._tmatrix_disconnected import stationary_distribution

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
    if eps_A is None:  # default transition probability, in order to avoid zero columns
        eps_A = 0.01 / nstates
    if eps_B is None:  # default output probability, in order to avoid zero columns
        eps_B = 0.01 / nfull

    # coarse-grain
    if nstates > n:
        raise NotImplementedError('Trying to initialize '+str(nstates)+'-state HMM from smaller '+str(n)+'-state MSM.')
    else:
        # pcca
        from msmtools.analysis.dense.pcca import PCCA
        pcca_obj = PCCA(Prev, nstates)
        M = pcca_obj.memberships

        # coarse-grained transition matrix
        A = coarse_grain_transition_matrix(P, M, eps=eps_A)

        # compress stationary probabilities.
        C_coarse = M.T.dot(C).dot(M)
        pi = stationary_distribution(C_coarse, A)  # need C_coarse in case if A is disconnected

        # full state space output matrix
        B = eps_B * np.ones((nstates, nfull), dtype=np.float64)
        # fill PCCA distributions if they exceed eps_B
        B[:, indexes] = np.maximum(B[:, indexes], pcca_obj.output_probabilities)
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:, None]

    return pi, A, B


def estimate_initial_hmm(observations, nstates, reversible=True, eps_A=None, eps_B=None):
    """Generate an initial HMM with discrete output densities

    Initialized HMM as described in [1]_. First estimates a Markov state model
    on the given observations, then uses PCCA+ to coarse-grain the transition
    matrix [2]_ which initializes the HMM transition matrix. The HMM output
    probabilities are given by Bayesian inversion from the PCCA+ memberships [1]_.

    A weak prior count matrix may be added in the first MSM estimation step, if
    the number of nontransient (closed) sets is smaller than the requested number
    of hidden states. The regularization parameters eps_A and eps_B are used
    to guarantee that the hidden transition matrix and output probability matrix
    have no zeros. HMM estimation algorithms such as the EM algorithm and the
    Bayesian sampling algorithm cannot recover from zero entries, i.e. once they
    are zero, they will stay zero.

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=int)
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    reversible : bool
        Estimate reversible HMM transition matrix.
    eps_A : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_B : float or None
        Minimum output probability. Default: 0.01 / nfull

    Raises
    ------
    NotImplementedError
        If the number of hidden states exceeds the number of observed states.

    Examples
    --------
    Generate initial model for a discrete output model.

    >>> from bhmm import testsystems
    >>> [model, observations, states] = testsystems.generate_synthetic_observations(output_model_type='discrete')
    >>> initial_model = estimate_initial_hmm(observations, model.nstates)

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
    C_full = msmtools.estimation.count_matrix(observations, 1).toarray()  # need to be dense here
    # truncate to states with at least one observed incoming or outgoing count.
    indexes = np.where(C_full.sum(axis=0) + C_full.sum(axis=1) > 0)[0]
    C = C_full[np.ix_(indexes, indexes)]
    # check if we can proceed
    if indexes.size < nstates:
        raise NotImplementedError('Trying to initialize ' + str(nstates) + '-state HMM from smaller '
                                  + str(indexes.size) + '-state MSM.')

    # MICROSTATE TRANSITION MATRIX (MSM). This matrix may be disconnected and have transient states
    P_msm = _tmatrix_disconnected.estimate_P(C, reversible=reversible)

    # MICROSTATE TRANSITION MATRIX FOR PCCA+
    # does the count matrix too few closed sets to give nstates metastable states? Then we need a prior
    if len(_tmatrix_disconnected.closed_sets(C)) < nstates:
        msm_prior = 0.001
        B = msm_prior * np.eye(C.shape[0])  # diagonal prior
        B += msmtools.estimation.prior_neighbor(C, alpha=msm_prior)  # neighbor prior
        C_post = C + B  # posterior
        P_for_pcca = _tmatrix_disconnected.estimate_P(C_post, reversible=True)
    elif reversible:  # in this case we already have a reversible estimate that is large enough
        P_for_pcca = P_msm
    else:  # enough metastable states, but not yet reversible. re-estimate
        P_for_pcca = _tmatrix_disconnected.estimate_P(C, reversible=True)

    # INITIALIZE HMM
    pi, A, B = msm_to_hmm(C, P_for_pcca, nstates, P=P_msm, indexes=indexes, nfull=C_full.shape[0],
                          eps_A=eps_A, eps_B=eps_B)
    if reversible:
        A = _tmatrix_disconnected.enforce_reversible_on_closed(A)

    #print 'cg pi: ', pi
    #print 'cg A:\n ', A
    #print 'cg B:\n ', B

    logger().info('Initial model: ')
    logger().info('transition matrix = \n'+str(A))
    logger().info('output matrix = \n'+str(B.T))

    # initialize HMM
    output_model = DiscreteOutputModel(B)
    model = HMM(pi, A, output_model)
    return model


