__author__ = 'noe'

import numpy as np

from bhmm.hmm_class import HMM

def initial_model_gaussian1d(observations, nstates, reversible=True, verbose=False):
    """Generate an initial model with 1D-Gaussian output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=float)
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    verbose : bool, optional, default=False
        If True, will be verbose in output.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> from bhmm import testsystems
    >>> [model, observations, states] = testsystems.generate_synthetic_observations(output_model_type='gaussian')
    >>> initial_model = initial_model_gaussian1d(observations, model.nstates)

    """
    ntrajectories = len(observations)

    # Concatenate all observations.
    collected_observations = np.array([], dtype=np.float64)
    for o_t in observations:
        collected_observations = np.append(collected_observations, o_t)

    # Fit a Gaussian mixture model to obtain emission distributions and state stationary probabilities.
    from sklearn import mixture
    gmm = mixture.GMM(n_components=nstates)
    gmm.fit(collected_observations)
    from bhmm import GaussianOutputModel
    output_model = GaussianOutputModel(nstates, means=gmm.means_[:,0], sigmas=np.sqrt(gmm.covars_[:,0]))

    if verbose:
        print "Gaussian output model:"
        print output_model

    # Extract stationary distributions.
    Pi = np.zeros([nstates], np.float64)
    Pi[:] = gmm.weights_[:]

    if verbose:
        print "GMM weights: %s" % str(gmm.weights_)

    # Compute transition matrix that gives specified Pi.
    Tij = np.tile(Pi, [nstates, 1])

    # Construct simple model.
    model = HMM(nstates, Tij, output_model)

    # Compute fractional state memberships.
    from scipy.misc import logsumexp
    Nij = np.zeros([nstates, nstates], np.float64)
    for o_t in observations:
        # length of trajectory
        T = o_t.shape[0]
        # output probability
        pobs = output_model.p_obs(o_t)
        # normalize
        pobs /= pobs.sum(axis=1)[:,None]
        # Accumulate fractional transition counts from this trajectory.
        for t in range(T-1):
            Nij[:,:] = Nij[:,:] + np.outer(pobs[t,:], pobs[t+1,:])

        if verbose:
            print "Nij"
            print Nij

    # Compute transition matrix maximum likelihood estimate.
    import pyemma.msm.estimation as msmest
    msmest.transition_matrix(Tij, reversible=reversible)

    # Update model.
    model = HMM(nstates, Tij, output_model, reversible=reversible)

    return model
