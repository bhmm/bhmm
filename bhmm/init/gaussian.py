__author__ = 'noe'

import numpy as np

from bhmm.hmm_class import HMM
import bhmm.msm.linalg as msmest

def initial_model_gaussian1d(observations, nstates, reversible = True):
    """Generate an initial model with 1D-Gaussian output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=float)
        list of arrays of length T_i with observation data

    TODO
    ----
    * Replace this with EM or MLHMM procedure from Matlab code.

    """
    ntrajectories = len(observations)

    # Concatenate all observations.
    collected_observations = np.array([], dtype=np.float64)
    for o_t in observations:
        collected_observations = np.append(collected_observations, o_t, axis=0)

    # Fit a Gaussian mixture model to obtain emission distributions and state stationary probabilities.
    from sklearn import mixture
    gmm = mixture.GMM(n_components=nstates)
    gmm.fit(collected_observations)
    from bhmm import GaussianOutputModel
    output_model = GaussianOutputModel(nstates, means=gmm.means_[:,0], sigmas=np.sqrt(gmm.covars_[:,0]))

    # DEBUG
    print "Gaussian output model:"
    print output_model

    # Extract stationary distributions.
    Pi = np.zeros([nstates], np.float64)
    Pi[:] = gmm.weights_[:]

    # DEBUG
    print "GMM weights: %s" % str(gmm.weights_)

    # Compute transition matrix that gives specified Pi.
    Tij = np.tile(Pi, [nstates, 1])

    # Construct simple model.
    model = HMM(nstates, Tij, output_model)

    # Compute fractional state memberships.
    from scipy.misc import logsumexp
    Nij = np.zeros([nstates, nstates], np.float64)
    for trajectory_index in range(ntrajectories):
        o_t = observations[trajectory_index] # extract trajectory
        T = o_t.shape[0]
        # Compute log emission probabilities.
        log_p_ti = np.zeros([T,nstates], np.float64)
        for i in range(nstates):
            log_p_ti[:,i] = model.log_emission_probability(i, o_t)
        # Exponentiate and normalize
        # TODO: Account for initial distribution.
        p_ti = np.zeros([T,nstates], np.float64)
        for t in range(T):
            p_ti[t,:] = np.exp(log_p_ti[t,:] - logsumexp(log_p_ti[t,:]))
            p_ti[t,:] /= p_ti[t,:].sum()
        print p_ti
        # Accumulate fractional transition counts from this trajectory.
        for t in range(T-1):
            Nij[:,:] = Nij[:,:] + np.outer(p_ti[t,:], p_ti[t+1,:])
        print "Nij"
        print Nij

    # Compute transition matrix maximum likelihood estimate.
    if (reversible):
        Tij = msmest.transition_matrix_MLE_reversible(Nij)
    else:
        Tij = msmest.transition_matrix_MLE_nonreversible(Nij)

    # Update model.
    model = HMM(nstates, Tij, output_model, reversible=reversible)

    return model