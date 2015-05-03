__author__ = 'noe'

from estimators.maximum_likelihood import MaximumLikelihoodEstimator as _MaximumLikelihoodEstimator
from estimators.bayesian_sampling import BHMM as _BHMM

def estimate_hmm(observations, nstates, initial_model=None, output_model_type='gaussian',
                 reversible=True, stationary=True, p=None, accuracy=1e-3, maxit=1000):
    r""" Estimate maximum-likelihood HMM

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    nstates : int
        The number of states in the model.
    initial_model : HMM, optional, default=None
        If specified, the given initial model will be used to initialize the BHMM.
        Otherwise, a heuristic scheme is used to generate an initial guess.
    output_model_type : str, optional, default='gaussian'
        Output model type.  ['gaussian', 'discrete']
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=True
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
    p : ndarray (nstates), optional, default=None
        Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
        estimated with the constraint that they have p as their stationary distribution. If given and
        stationary=False, p is the fixed initial distribution of hidden states.
    accuracy : float
        convergence threshold for EM iteration. When two the likelihood does not increase by more than accuracy, the
        iteration is stopped successfully.
    maxit : int
        stopping criterion for EM iteration. When so many iterations are performanced without reaching the requested
        accuracy, the iteration is stopped without convergence (a warning is given)

    Return
    ------
    hmm : :class:`HMM <bhmm.hmm.generic_hmm.HMM>`

    """
    # construct estimator
    est = _MaximumLikelihoodEstimator(observations, nstates, initial_model=None, output_model_type=output_model_type,
                                      reversible=True, stationary=True, p=None, accuracy=1e-3, maxit=1000)
    # run
    est.fit()
    # return model
    return est.hmm


def bayesian_hmm(observations, estimated_hmm, nsample=100, store_hidden=False):
    r""" Bayesian HMM based on sampling the posterior

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    estimated_hmm : HMM
        HMM estimated from estimate_hmm or initialize_hmm
    nsample : int, optional, default=100
        number of Gibbs sampling steps
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs

    Return
    ------
    hmm : :class:`SampledHMM <bhmm.hmm.generic_sampled_hmm.SampledHMM>`

    """
    # construct estimator
    sampler = _BHMM(observations, estimated_hmm.nstates, initial_model=estimated_hmm,
                    reversible=estimated_hmm.is_reversible, transition_matrix_sampling_steps=1000,
                    output_model_type=estimated_hmm.output_model.model_type)

    # Sample models.
    sampled_hmms = sampler.sample(nsamples=nsample, save_hidden_state_trajectory=store_hidden)
    # return model
    from bhmm.hmm.generic_sampled_hmm import SampledHMM
    return SampledHMM(estimated_hmm, sampled_hmms)
