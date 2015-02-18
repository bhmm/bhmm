"""
Test systems for validation

"""

import numpy as np
from scipy import linalg

#from bhmm import HMM
import math

def generate_transition_matrix(nstates = 3, lifetime_max = 10000, lifetime_min = 100, reversible = True):
    """
    Generates random metastable transition matrices

    Parameters
    ----------
    nstates : int, optional, default=3
        Number of states for which row-stockastic transition matrix is to be generated.
    lifetime_max : float, optional, default = 10000
        maximum lifetime of any state
    lifetime_min : float, optional, default = 100
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
    states.append({ 'model' : 'gaussian', 'mu' : -1, 'sigma' : sigma })
    states.append({ 'model' : 'gaussian', 'mu' :  0, 'sigma' : sigma })
    states.append({ 'model' : 'gaussian', 'mu' : +1, 'sigma' : sigma })

    Tij = generate_transition_matrix(nstates, reversible=True)

    # Construct HMM with these parameters.
    model = HMM(nstates, Tij, states)

    return model


def generate_random_model(nstates, reversible=True):
    """Generate a random HMM model with the specified number of states.

    Parameters
    ----------
    nstates : int
        The number of states for the model.
    reversible : bool, optional, default=True
        If True, the row-stochastic transition matrix will be reversible.

    Returns
    -------
    model : HMM
        The randomly generated HMM model.

    """
    Tij = generate_transition_matrix(nstates, reversible=reversible)

    states = list()
    for index in range(nstates):
        # TODO: Come up with a better prior with tunable hyperparameters on mu and sigma.
        mu = np.random.randn()
        sigma = np.random.random()
        states.append({ 'model' : 'gaussian', 'mu' : mu, 'sigma' : sigma })

    # Construct HMM with these parameters.
    model = HMM(nstates, Tij, states)

    return model


def generate_random_bhmm(nstates=3, ntrajectories=10, length=100):
    """Generate a BHMM model from synthetic data from a random HMM model.

    Parameters
    ----------
    nstates : int, optional, default=3
        The number of states for the underlying HMM model.
    ntrajectories : int, optional, default=10
        The number of synthetic observation trajectories to generate.
    length : int, optional, default=100
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
    model = generate_random_model(nstates)
    # Generate synthetic data.
    observations = model.generate_synthetic_observation_trajectories(ntrajectories=ntrajectories, length=length)
    # Initialize a new BHMM model.
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
