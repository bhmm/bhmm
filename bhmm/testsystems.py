"""
Test systems for validation

"""

import numpy as np
from scipy import linalg

from hmm import HMM

def transition_matrix(nstates=3):
    """
    Construct a test transition matrix.

    Returns
    -------
    Tij : np.array with shape (nstates, nstates)
        The transition matrix.

    """
    if nstates != 3:
        raise Exception("Only 3 states are supported right now.")

    # Define row-stochastic rate matrix that satisfies detailed balance, and compute transition matrix from this.
    Kij = np.array([[-0.10,  0.10,  0.00],
                     [ 0.10, -0.15,  0.05],
                     [ 0.00,  0.05, -0.05]], np.float64)
    Tij = linalg.expm(Kij);

    return Tij

def three_state_model(sigma=1.0):
    """
    Construct a test three-state model with variable emission width.

    Parameters
    ----------
    sigma : float
        The width of the observed gaussian distribution for each state.

    """
    nstates = 3

    # Define state emission probabilities.
    states = list()
    states.append({ 'mu' : -1, 'sigma' : sigma })
    states.append({ 'mu' :  0, 'sigma' : sigma })
    states.append({ 'mu' : +1, 'sigma' : sigma })

    Tij = transition_matrix(nstates)

    # Construct HMM with these parameters.
    model = HMM(nstates, Tij, states)

    return model

