__author__ = 'noe'

import numpy as np
from bhmm.hmm_class import HMM
from bhmm.output_models.discrete import DiscreteOutputModel

import warnings

def initial_model_discrete(observations, nstates, lag=1, reversible=True, verbose=False):
    """Generate an initial model with discrete output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=int)
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    lag : int, optional, default=1
        The lag time to use for initializing the model.
    verbose : bool, optional, default=False
        If True, will be verbose in output.

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

    # import emma inside function in order to avoid dependency loops
    import pyemma.msm.estimation as msmest
    import pyemma.msm.analysis as msmana
    from pyemma.msm.analysis.dense import pcca

    # check input
    if not reversible:
        warnings.warn("nonreversible initialization of discrete HMM currently not supported. Using a reversible matrix for initialization.")
        reversible = True

    # estimate msm
    # ------------
    # count matrix
    C = msmest.count_matrix(observations, lag).toarray()
    #print 'C = \n',C

    nmicro = C.shape[0]
    # connected count matrix
    C_conn = msmest.connected_cmatrix(C)
    giant = msmest.largest_connected_set(C)

    # transition matrix
    P = msmest.transition_matrix(C_conn, reversible=reversible)
    #print 'P = \n',P

    # pcca coarse-graining
    # --------------------
    # PCCA memberships
    chi = pcca.pcca(P, nstates)
    #print 'chi = \n',chi
    # stationary distribution
    pi = msmana.stationary_distribution(P)
    # coarse-grained stationary distribution
    pi_coarse = np.dot(chi.T, pi)
    #print 'pi_coarse = \n',pi_coarse

    # HMM output matrix
    B_conn = np.dot(np.dot(np.diag(1.0/pi_coarse), chi.T), np.diag(pi))

    #print 'B_conn = \n',B_conn
    # full state space output matrix
    eps = 0.01 * (1.0/nmicro) # default output probability, in order to avoid zero columns
    B = eps * np.ones((nstates,nmicro), dtype=np.float64)
    # expand B_conn to full state space
    B[:,giant] = B_conn[:,:]

    # DE-MIX populations: Assign all output probability to the largest contributor.
    # This step is a bit made-up because (A,B) is now not a self-consistent PCCA pair anymore - however it works much better.
    sums = B.sum(axis=0)
    amax = B.argmax(axis=0)
    B[:,:] = 0.0
    for i in range(B.shape[1]):
        B[amax[i],i] = sums[i]
    # renormalize B to make it row-stochastic
    B /= B.sum(axis=1)[:,None]


    # coarse-grained transition matrix
    A = pcca.coarsegrain(P, nstates)
    # renormalize to eliminate numerical errors
    A /= A.sum(axis=1)[:,None]

    if verbose:
        print 'Initial model: '
        print 'A = \n',A
        print 'B.T = \n'
        for i in range(B.shape[1]):
            print B[0,i],B[1,i]
        print

    # initialize HMM
    # --------------
    output_model = DiscreteOutputModel(B)
    model = HMM(nstates, A, output_model)
    return model


