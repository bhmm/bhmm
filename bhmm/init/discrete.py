__author__ = 'noe'

import numpy as np
from bhmm.hmm_class import HMM
from bhmm.output_models.discrete import DiscreteOutputModel

import warnings

def initial_model_discrete(observations, nstates, lag = 1, reversible = True):
    """Generate an initial model with discrete output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=int)
        list of arrays of length T_i with observation data

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

    nmicro = C.shape[0]
    # connected count matrix
    C_conn = msmest.connected_cmatrix(C)
    giant = msmest.largest_connected_set(C)

    # transition matrix
    P = msmest.transition_matrix(C_conn, reversible=reversible)

    # pcca coarse-graining
    # --------------------
    # PCCA memberships
    chi = pcca.pcca(P, nstates)
    # stationary distribution
    pi = msmana.stationary_distribution(P)
    # coarse-grained stationary distribution
    pi_coarse = np.dot(chi.T, pi)
    # HMM output matrix
    B_conn = np.dot(np.dot(np.diag(1.0/pi_coarse), chi.T), np.diag(pi))
    # expand B to full state space
    B = np.zeros((nstates,nmicro))
    B[:,giant] = B_conn[:,:]
    # coarse-grained transition matrix
    A = pcca.coarsegrain(P, nstates)

    # initialize HMM
    # --------------
    output_model = DiscreteOutputModel(B)
    model = HMM(nstates, A, output_model)
    return model


