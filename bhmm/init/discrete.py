__author__ = 'noe'

import warnings

import numpy as np

from bhmm.hmm.generic_hmm import HMM
from bhmm.output_models.discrete import DiscreteOutputModel
from bhmm.util.logger import logger


def initial_model_discrete(observations, nstates, lag=1, reversible=True):
    """Generate an initial model with discrete output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=int)
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    lag : int, optional, default=1
        The lag time to use for initializing the model.

    TODO
    ----
    * Why do we have a `lag` option?  Isn't the HMM model, by definition, lag=1 everywhere?  Why would this be useful instead of just having the user subsample the data?

    Examples
    --------

    Generate initial model for a discrete output model.

    >>> from bhmm import testsystems
    >>> [model, observations, states] = testsystems.generate_synthetic_observations(output_model_type='discrete')
    >>> initial_model = initial_model_discrete(observations, model.nstates)

    """
    # check input
    if not reversible:
        warnings.warn("nonreversible initialization of discrete HMM currently not supported. Using a reversible matrix for initialization.")
        reversible = True

    # import emma inside function in order to avoid dependency loops
    from pyemma import msm

    # estimate Markov model
    MSM = msm.estimate_markov_model(observations, lag, reversible=True, connectivity='largest')

    # PCCA
    pcca = MSM.pcca(nstates)

    # HMM output matrix
    B_conn = MSM.metastable_distributions

    #print 'B_conn = \n',B_conn
    # full state space output matrix
    nstates_full = MSM.count_matrix_full.shape[0]
    eps = 0.01 * (1.0/nstates_full) # default output probability, in order to avoid zero columns
    B = eps * np.ones((nstates,nstates_full), dtype=np.float64)
    # expand B_conn to full state space
    B[:,MSM.active_set] = B_conn[:,:]
    # renormalize B to make it row-stochastic
    B /= B.sum(axis=1)[:,None]

    # coarse-grained transition matrix
    M = pcca.memberships
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, MSM.transition_matrix), M)
    P_coarse = np.dot(W, A)

    # symmetrize and renormalize to eliminate numerical errors
    X = np.dot(np.diag(pcca.coarse_grained_stationary_probability), P_coarse)
    X = 0.5 * (X + X.T)
    # if there are values < 0, set to eps
    X = np.maximum(X, eps)
    # turn into coarse-grained transition matrix
    A = X / X.sum(axis=1)[:, None]

    logger().info('Initial model: ')
    logger().info('transition matrix = \n'+str(A))
    logger().info('output matrix = \n'+str(B.T))

    # initialize HMM
    # --------------
    output_model = DiscreteOutputModel(B)
    model = HMM(A, output_model)
    return model


