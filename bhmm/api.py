__author__ = 'noe'

import numpy as _np


# TODO: type needs to be renamed because it shadows built-in type

def _guess_model_type(observations):
    """ Suggests a HMM model type based on the observation data

    Uses simple rules in order to decide which HMM model type makes sense based on observation data.
    If observations consist of arrays/lists of integer numbers (irrespective of whether the python type is
    int or float), our guess is 'discrete'.
    If observations consist of arrays/lists of 1D-floats, our guess is 'discrete'.
    In any other case, a TypeError is raised because we are not supporting that data type yet.

    Parameters:
    -----------
    observations : list of lists or arrays
        observation trajectories

    Returns:
    --------
    model_type : str
        One of {'discrete', 'gaussian'}

    """
    from bhmm.util import types as _types

    o1 = _np.array(observations[0])

    # CASE: vector of int? Then we want a discrete HMM
    if _types.is_int_vector(o1):
        return 'discrete'

    # CASE: not int type, but everything is an integral number. Then we also go for discrete
    if _np.allclose(o1, _np.round(o1)):
        isintegral = True
        for i in range(1, len(observations)):
            if not _np.allclose(observations[i], _np.round(observations[i])):
                isintegral = False
                break
        if isintegral:
            return 'discrete'

    # CASE: vector of double? Then we want a gaussian
    if _types.is_float_vector(o1):
        return 'gaussian'

    # None of the above? Then we currently do not support this format!
    raise TypeError('Observations is neither sequences of integers nor 1D-sequences of floats. The current version'
                    'does not support your input.')


def lag_observations(observations, lag, stride=1):
    r""" Create new trajectories that are subsampled at lag but shifted

    Given a trajectory (s0, s1, s2, s3, s4, ...) and lag 3, this function will generate 3 trajectories
    (s0, s3, s6, ...), (s1, s4, s7, ...) and (s2, s5, s8, ...). Use this function in order to parametrize a MLE
    at lag times larger than 1 without discarding data. Do not use this function for Bayesian estimators, where
    data must be given such that subsequent transitions are uncorrelated.

    Parameters
    ----------
    observations : list of int arrays
        observation trajectories
    lag : int
        lag time
    stride : int, default=1
        will return only one trajectory for every stride. Use this for Bayesian analysis.

    """
    obsnew = []
    for obs in observations:
        for shift in range(0, lag, stride):
            obs_lagged = (obs[shift:][::lag])
            if len(obs_lagged) > 1:
                obsnew.append(obs_lagged)
    return obsnew


def init_hmm(observations, nstates, lag=1, type=None, reversible=True):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    type : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output_model_type='gaussian')
    >>> initial_model = init_hmm(observations, model.nstates, type='gaussian')

    Generate initial model for a discrete output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output_model_type='discrete')
    >>> initial_model = init_hmm(observations, model.nstates, type='discrete')

    """
    # select output model type
    if type is None:
        type = _guess_model_type(observations)

    if type == 'discrete':
        return init_discrete_hmm(observations, nstates, lag=lag, reversible=reversible)
    elif type == 'gaussian':
        return init_gaussian_hmm(observations, nstates, lag=lag, reversible=reversible)
    else:
        raise NotImplementedError('output model type '+str(type)+' not yet implemented.')


def init_gaussian_hmm(observations, nstates, lag=1, reversible=True):
    """ Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output_model_type='gaussian')
    >>> initial_model = init_gaussian_hmm(observations, model.nstates)

    """
    from bhmm.init import gaussian
    if lag > 1:
        observations = lag_observations(observations, lag)
    hmm0 = gaussian.initial_model_gaussian1d(observations, nstates, reversible=reversible)
    hmm0._lag = lag
    return hmm0


def init_discrete_hmm(observations, nstates, lag=1, reversible=True,
                      msm_neighbor_prior=0.001, msm_diag_prior=0.001,
                      eps_P=0.001, eps_pout=0.001, separate=None):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    msm_neighbor_prior : float
        prior used to strongly connect weakly connected sets before MSM estimation.
        Defined by:
        * msm_neighbor_prior if c_ij + c_ji > 0
        * 0 else.
    msm_diag_prior : float
        will add msm_diag_prior to all nonempty states i (defined by :math:`\sum_j c_ij + c_ji > 0`)
    eps_P : float or None
        Minimum transition probability. Default: 0.01 / nstates
    eps_pout : float or None
        Minimum output probability. Default: 0.01 / nfull
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.

    Examples
    --------

    Generate initial model for a discrete output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output_model_type='discrete')
    >>> initial_model = init_discrete_hmm(observations, model.nstates)

    """
    from bhmm.init.discrete import estimate_initial_hmm
    from bhmm.estimators import _tmatrix_disconnected
    import msmtools.estimation as msmest
    C = msmest.count_matrix(observations, lag).toarray()
    if msm_neighbor_prior > 0:
        C += msmest.prior_neighbor(C, msm_neighbor_prior)
    if msm_diag_prior > 0:
        nonempty = _tmatrix_disconnected.nonempty_set(C)
        C[nonempty, nonempty] += msm_diag_prior
    hmm0 = estimate_initial_hmm(C, nstates, reversible=reversible, eps_A=eps_P, eps_B=eps_pout, separate=separate)
    hmm0._lag = lag
    return hmm0


def gaussian_hmm(pi, P, means, sigmas):
    """ Initializes a 1D-Gaussian HMM

    Parameters
    ----------
    pi : ndarray(nstates, )
        Initial distribution.
    P : ndarray(nstates,nstates)
        Hidden transition matrix
    means : ndarray(nstates, )
        Means of Gaussian output distributions
    sigmas : ndarray(nstates, )
        Standard deviations of Gaussian output distributions
    stationary : bool, optional, default=True
        If True: initial distribution is equal to stationary distribution of transition matrix
    reversible : bool, optional, default=True
        If True: transition matrix will fulfill detailed balance constraints.

    """
    from bhmm.hmm.gaussian_hmm import GaussianHMM
    from bhmm.output_models.gaussian import GaussianOutputModel
    # count states
    nstates = _np.array(P).shape[0]
    # initialize output model
    output_model = GaussianOutputModel(nstates, means, sigmas)
    # initialize general HMM
    from bhmm.hmm.generic_hmm import HMM as _HMM
    ghmm = _HMM(pi, P, output_model)
    # turn it into a Gaussian HMM
    ghmm = GaussianHMM(ghmm)
    return ghmm


def discrete_hmm(pi, P, pout):
    """ Initializes a discrete HMM

    Parameters
    ----------
    pi : ndarray(nstates, )
        Initial distribution.
    P : ndarray(nstates,nstates)
        Hidden transition matrix
    pout : ndarray(nstates,nsymbols)
        Output matrix from hidden states to observable symbols
    pi : ndarray(nstates, )
        Fixed initial (if stationary=False) or fixed stationary distribution (if stationary=True).
    stationary : bool, optional, default=True
        If True: initial distribution is equal to stationary distribution of transition matrix
    reversible : bool, optional, default=True
        If True: transition matrix will fulfill detailed balance constraints.

    """
    from bhmm.hmm.discrete_hmm import DiscreteHMM
    from bhmm.output_models.discrete import DiscreteOutputModel
    # initialize output model
    output_model = DiscreteOutputModel(pout)
    # initialize general HMM
    from bhmm.hmm.generic_hmm import HMM as _HMM
    dhmm = _HMM(pi, P, output_model)
    # turn it into a Gaussian HMM
    dhmm = DiscreteHMM(dhmm)
    return dhmm


def estimate_hmm(observations, nstates, lag=1, initial_model=None, type=None,
                 reversible=True, stationary=False, p=None, accuracy=1e-3, maxit=1000, maxit_P=100000,
                 mincount_connectivity=1e-2):
    r""" Estimate maximum-likelihood HMM

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    nstates : int
        The number of states in the model.
    lag : int
        the lag time at which observations should be read
    initial_model : HMM, optional, default=None
        If specified, the given initial model will be used to initialize the BHMM.
        Otherwise, a heuristic scheme is used to generate an initial guess.
    type : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    p : ndarray (nstates), optional, default=None
        Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
        estimated with the constraint that they have p as their stationary distribution. If given and
        stationary=False, p is the fixed initial distribution of hidden states.
    accuracy : float
        convergence threshold for EM iteration. When two the likelihood does not increase by more than accuracy, the
        iteration is stopped successfully.
    maxit : int
        stopping criterion for EM iteration. When so many iterations are performed without reaching the requested
        accuracy, the iteration is stopped without convergence (a warning is given)

    Return
    ------
    hmm : :class:`HMM <bhmm.hmm.generic_hmm.HMM>`

    """
    # select output model type
    if type is None:
        type = _guess_model_type(observations)

    if lag > 1:
        observations = lag_observations(observations, lag)

    # construct estimator
    from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as _MaximumLikelihoodEstimator
    est = _MaximumLikelihoodEstimator(observations, nstates, initial_model=initial_model, type=type,
                                      reversible=reversible, stationary=stationary, p=p, accuracy=accuracy,
                                      maxit=maxit, maxit_P=maxit_P)
    # run
    est.fit()
    # set lag time
    est.hmm._lag = lag
    # return model
    return est.hmm


def bayesian_hmm(observations, estimated_hmm, nsample=100, reversible=True, stationary=False,
                 p0_prior='mixed', transition_matrix_prior='mixed', store_hidden=False, call_back=None):
    r""" Bayesian HMM based on sampling the posterior

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    estimated_hmm : HMM
        HMM estimated from estimate_hmm or initialize_hmm
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the stationary distribution of the transition matrix will be used as initial distribution.
        Only use True if you are confident that the observation trajectories are started from a global
        equilibrium. If False, the initial distribution will be estimated as usual from the first step
        of the hidden trajectories.
    nsample : int, optional, default=100
        number of Gibbs sampling steps
    p0_prior : None or float or ndarray(n)
        prior count array for the initial distribution to be used for transition matrix sampling.
        |  'mixed' (default),  1 count is distributed according to p0 of initial model
        |  None,  -1 prior is used that ensures coincidence between mean an MLE.
            Will sooner or later lead to sampling problems, because as soon as zero trajectories are drawn
            from a given state, the sampler cannot recover and that state will never serve as a starting
            state subsequently. Only recommended for when the probability to sample zero trajectories
            from any state is negligible.
    transition_matrix_prior : str or ndarray(n,n)
        prior count matrix to be used for transition matrix sampling, or a keyword specifying the prior mode
        |  'mixed' (default),  1 count is distributed to every row according to P of initial model.
        |  None,  -1 prior is used that ensures coincidence between mean and MLE. Can lead to sampling
            disconnected matrices in the low-data regime. If you have disconnectivity problems, consider
            using 'init-connect'
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs
    call_back : function, optional, default=None
        a call back function with no arguments, which if given is being called
        after each computed sample. This is useful for implementing progress bars.

    Return
    ------
    hmm : :class:`SampledHMM <bhmm.hmm.generic_sampled_hmm.SampledHMM>`

    """
    # construct estimator
    from bhmm.estimators.bayesian_sampling import BayesianHMMSampler as _BHMM
    sampler = _BHMM(observations, estimated_hmm.nstates, initial_model=estimated_hmm,
                    reversible=reversible, stationary=stationary, transition_matrix_sampling_steps=1000,
                    p0_prior=p0_prior, transition_matrix_prior=transition_matrix_prior,
                    type=estimated_hmm.output_model.model_type)

    # Sample models.
    sampled_hmms = sampler.sample(nsamples=nsample, save_hidden_state_trajectory=store_hidden,
                                  call_back=call_back)
    # return model
    from bhmm.hmm.generic_sampled_hmm import SampledHMM
    return SampledHMM(estimated_hmm, sampled_hmms)
