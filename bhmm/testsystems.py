"""
Test systems for validation

"""

import numpy as np
from scipy import linalg

#from bhmm import HMM
import math

def generate_transition_matrix(nstates=3, lifetime_max=100, lifetime_min=10, reversible=True):
    """
    Generates random metastable transition matrices

    Parameters
    ----------
    nstates : int, optional, default=3
        Number of states for which row-stockastic transition matrix is to be generated.
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    reversible : bool, optional, default=True
        If True, the row-stochastic transition matrix will be reversible.

    Returns
    -------
    Tij : np.array with shape (nstates, nstates)
        A randomly generated row-stochastic transition matrix.

    """
    # regular grid in the log lifetimes
    ltmax = math.log(lifetime_max)
    ltmin = math.log(lifetime_min)
    lt = np.linspace(ltmin, ltmax, num = nstates)
    # create diagonal with self-transition probabilities according to timescales
    diag = 1.0 - 1.0/np.exp(lt)
    # random X
    X = np.random.random((nstates,nstates))
    if (reversible):
        X += X.T
    # row-normalize
    T = X / np.sum(X, axis=1)[:,None]
    # enforce lifetimes by rescaling rows
    for i in range(nstates):
        T[i,i] = 0
        T[i,:] *= (1.0-diag[i]) / np.sum(T[i,:])
        T[i,i] = 1.0 - np.sum(T[i,:])

    return T


def dalton_model(nstates = 3, omin = -1, omax = 1, sigma_min = 0.5, sigma_max = 2.0, lifetime_max = 10000, lifetime_min = 100, reversible = True):
    """
    Construct a test two-state model with regular spaced emission means (linearly interpolated between omin and omax)
    and variable emission widths (linearly interpolated between sigma_min and sigma_max).

    Parameters
    ----------
    nstates : int, optional, default = 3
        number of hidden states
    omin : float, optional, default = -1
        mean position of the first state.
    omax : float, optional, default = 1
        mean position of the last state.
    sigma_min : float, optional, default = 0.5
        The width of the observed gaussian distribution for the first state
    sigma_max : float, optional, default = 2.0
        The width of the observed gaussian distribution for the last state
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    reversible : bool, optional, default=True
        If True, the row-stochastic transition matrix will be reversible.

    """
    nstates = 3

    # parameters
    means = np.linspace(omin, omax, num = nstates)
    sigmas = np.linspace(sigma_min, sigma_max, num = nstates)

    # Define state emission probabilities.
    states = list()
    for i in range(nstates):
        states.append({ 'model' : 'gaussian', 'mu' : means[i], 'sigma' : sigmas[i] })

    Tij = generate_transition_matrix(nstates, lifetime_max = lifetime_max, lifetime_min = lifetime_min, reversible = reversible)

    # Construct HMM with these parameters.
    from bhmm import HMM
    model = HMM(nstates, Tij, states)

    return model


def generate_random_bhmm(nstates=3, ntrajectories=10, length=100,
                         lifetime_max = 10000, lifetime_min = 100, reversible = True):
    """Generate a BHMM model from synthetic data from a random HMM model.

    Parameters
    ----------
    nstates : int, optional, default=3
        The number of states for the underlying HMM model.
    ntrajectories : int, optional, default=10
        The number of synthetic observation trajectories to generate.
    length : int, optional, default=10000
        The length of synthetic observation trajectories to generate.

    Returns
    -------
    model : HMM
        The true underlying HMM model.
    observations : list of numpy arrays
        The synthetic observation trajectories generated from the HMM model.
    bhmm : BHMM
        The BHMM model generated.

    Examples
    --------

    Generate BHMM with default parameters.

    >>> [model, observations, bhmm] = generate_random_bhmm()

    """

    # Generate a random HMM model.
    model = dalton_model(nstates, lifetime_max = lifetime_max, lifetime_min = lifetime_min, reversible = reversible)
    # Generate synthetic data.
    observations = model.generate_synthetic_observation_trajectories(ntrajectories=ntrajectories, length=length)
    # Initialize a new BHMM model.
    from bhmm import BHMM
    bhmm = BHMM(observations, nstates)

    return [model, observations, bhmm]


def main():
    """
    This is a test function

    :return:
    """
    T = generate_transition_matrix()
    eigs = np.linalg.eigvals(T)
    eigs = np.sort(eigs)[1::-1]
    ts = -1.0 / np.log(eigs)
    print ts



if __name__ == "__main__":
    main()
