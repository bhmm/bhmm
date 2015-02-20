"""Python implementation of Hidden Markov Model kernel functions

This module is considered to be the reference for checking correctness of other
kernels. All implementations are being kept very simple, straight forward and
closely related to Rabiners [1] paper.

.. [1] Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models and
   Selected Applications in Speech Recognition", Proceedings of the IEEE,
   vol. 77, issue 2
"""
import numpy as np


def forward(A, pobs, pi, dtype=np.float32):
    """Compute P(ob|A,B,pi) and all forward coefficients. With scaling!

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    pobs : numpy.array of floating numbers and shape (T,N)
        symbol probability matrix for each time and hidden state
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    prob : floating number
           The probability to observe the sequence `ob` with the model given
           by `A`, `B` and `pi`.
    alpha : np.array of floating numbers and shape (T,N)
            alpha[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.
    scaling : np.array of floating numbers and shape (T)
            scaling factors for each step in the calculation. can be used to
            rescale backward coefficients.

    See Also
    --------
    forward_no_scaling : Compute forward coefficients without scaling
    """
    T, N = pobs.shape[0], len(A)
    alpha = np.zeros((T,N), dtype=dtype)
    scale = np.zeros(T, dtype=dtype)

    # initial values
    # alpha_i(0) = pi_i * B_i,ob[0]
    alpha[0,:] = np.multiply(pi, pobs[0,:])
    # scaling factor
    scale[0] = np.sum(alpha[0,:])
    # scale
    alpha[0,:] /= scale[0]

    # induction
    for t in range(T-1):
        # alpha_j(t+1) = sum_i alpha_i(t) * A_i,j * B_j,ob(t+1)
        alpha[t+1,:] = np.dot(alpha[t,:], A) * pobs[t+1,:]
        # scaling factor
        scale[t+1] = np.sum(alpha[t+1,:])
        # scale
        alpha[t+1,:] /= scale[t+1]

    # log-likelihood
    logprob = 0.0
    for t in range(T):
        logprob += np.log(scale[t])
    return (logprob, alpha, scale)


def backward(A, pobs, dtype=np.float32):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    pobs : numpy.array of floating numbers and shape (T,N)
        symbol probability matrix for each time and hidden state
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    beta : np.array of floating numbers and shape (T,N)
            beta[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.

    See Also
    --------
    backward_no_scaling : Compute backward coefficients without scaling
    """
    T, N = pobs.shape[0], len(A)
    beta = np.zeros((T,N), dtype=dtype)
    scale = np.zeros(T, dtype=dtype)

    # initialization
    beta[T-1,:] = 1.0
    # scaling factor
    scale[T-1] = np.sum(beta[T-1,:])
    # scale
    beta[T-1,:] /= scale[T-1]

    # induction
    for t in range(T-2, -1, -1):
        # beta_i(t) = sum_j A_i,j * beta_j(t+1) * B_j,ob(t+1)
        beta[t,:] = np.dot(A, beta[t+1,:] * pobs[t+1,:])
        if pobs[t+1,0] == 0:
            raise ValueError('found 0s in t='+str(t)+' pobs[t+1] = '+str(pobs[t+1]))
        # scaling factor
        scale[t] = np.sum(beta[t,:])
        # scale
        beta[t,:] /= scale[t]
    return beta


def update(gamma, xi, ob, M, dtype=np.float32):
    """ Return an updated model for given state and transition counts.

    Parameters
    ----------
    gamma : numpy.array shape (T,N)
            state probabilities for each t
    xi : numpy.array shape (T,N,N)
         transition probabilities for each t
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)

    Returns
    -------
    A : numpy.array (N,N)
        new transition matrix
    B : numpy.array (N,M)
        new symbol probabilities
    pi : numpy.array (N)
         new initial distribution
    dtype : { nupmy.float64, numpy.float32 }, optional

    Notes
    -----
    This function is part of the Baum-Welch algorithm for a single observation.

    See Also
    --------
    state_probabilities : to calculate `gamma`
    transition_probabilities : to calculate `xi`

    """
    T,N = len(ob), len(gamma[0])
    # starting distribution
    pi = gamma[0,:]
    # all but the last time step
    gamma_sum = np.sum(gamma, axis=0) - gamma[T-1,:]
    # new transition matrix
    A = np.sum(xi, axis=0) / gamma_sum
    # add the last time step
    gamma_sum += gamma[T-1,:]
    # output probabilities
    B  = np.zeros((N,M), dtype=dtype)
    for k in range(M):
        times = np.where(ob == k)[0]
        B[:,k] = np.sum(gamma[times,:], axis=0) / gamma_sum
    return (A, B, pi)


def state_probabilities(alpha, beta, dtype=np.float32):
    """ Calculate the (T,N)-probabilty matrix for being in state i at time t.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    dtype : item datatype [optional]

    Returns
    -------
    gamma : numpy.array shape (T,N)
            gamma[t,i] is the probabilty at time t to be in state i !

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    T, N = len(alpha), len(alpha[0])
    gamma = np.zeros((T,N), dtype=dtype)
    for t in range(T):
        # gamma_i(t) = alpha_i(t) * beta_i(t)
        gamma[t,:] = alpha[t,:] * beta[t,:]
        # normalize
        gamma[t,:] /= np.sum(gamma[t,:])
    return gamma


def state_counts(gamma, T, dtype=np.float32):
    """ Sum the probabilities of being in state i to time t

    Parameters
    ----------
    gamma : numpy.array shape (T,N)
            gamma[t,i] is the probabilty at time t to be in state i !
    T : number of observationsymbols
    dtype : item datatype [optional]

    Returns
    -------
    count : numpy.array shape (N)
            count[i] is the summed probabilty to be in state i !

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    return np.sum(gamma[0:T], axis=0)


def symbol_counts(gamma, ob, M, dtype=np.float32):
    """ Sum the observed probabilities to see symbol k in state i.

    Parameters
    ----------
    gamma : numpy.array shape (T,N)
            gamma[t,i] is the probabilty at time t to be in state i !
    ob : numpy.array shape (T)
    M : integer. number of possible observationsymbols
    dtype : item datatype, optional

    Returns
    -------
    counts : numpy.array shape (N,M)

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    T, N = len(gamma), len(gamma[0])
    counts = np.zeros((N,M), dtype=type)
    for t in range(T):
        for i in range(N):
            counts[i,ob[t]] += gamma[t,i]
    return counts


def transition_probabilities(alpha, beta, A, pobs, dtype=np.float32):
    """ Compute for each t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    A : numpy.array shape (N,N)
        transition matrix of the model
    pobs : numpy.array of floating numbers and shape (T,N)
        symbol probability matrix for each time and hidden state
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)
    dtype : item datatype [optional]

    Returns
    -------
    xi : numpy.array shape (T-1, N, N)
         xi[t, i, j] is the probability to transition from i to j at time t.

    Notes
    -----
    It does not matter if alpha or beta scaled or not, as long as there scaling
    does not depend on the second variable.

    See Also
    --------
    state_counts : calculate the probability to be in state i at time t
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    T, N = pobs.shape[0], len(A)
    xi = np.zeros((T-1,N,N), dtype=dtype)
    for t in range(T-1):
        # xi_i,j(t) = alpha_i(t) * A_i,j * B_j,ob(t+1) * beta_j(t+1)
        xi[t,:,:] = np.dot(alpha[t,:][:,None] * A, np.diag(pobs[t+1,:] * beta[t+1,:]))
        # normalize to 1 for each time step
        xi[t,:,:] /= np.sum(xi[t,:,:])
    return xi


def transition_counts(alpha, beta, A, pobs, dtype=np.float32):
    """ Sum for all t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    A : numpy.array shape (N,N)
        transition matrix of the model
    pobs : numpy.array of floating numbers and shape (T,N)
        symbol probability matrix for each time and hidden state
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)
    dtype : item datatype [optional]

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j
         int time [0,T)

    Notes
    -----
    It does not matter if alpha or beta scaled or not, as long as there scaling
    does not depend on the second variable.

    See Also
    --------
    transition_probabilities : return the matrix of transition probabilities
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`
    """
    #TODO: this function is not needed. Counting should be done in transition probabilities.
    xi = transition_probabilities(alpha, beta, A, pobs, dtype=dtype)
    return np.sum(xi, axis=0)

def random_sequence(A, B, pi, T):
    """ Generate an observation sequence of length T from the model A, B, pi.

    Parameters
    ----------
    A : numpy.array shape (N,N)
        transition matrix of the model
    B : numpy.array shape (N,M)
        symbol probability matrix of the model
    pi : numpy.array shape (N)
         starting probability vector of the model
    T : integer
        length of generated observation sequence

    Returns
    -------
    obs : numpy.array shape (T)
          observation sequence containing only symbols, i.e. ints in [0,M)

    Notes
    -----
    This function relies on the function draw_state(distr).

    See Also
    --------
    draw_state : draw the index of the state, obeying the probability
                 distribution vector distr

    """
    obs    = np.zeros(T, dtype=np.int16)
    state  = draw_state(pi)
    obs[0] = state
    for t in range(1,T):
        state  = draw_state( A[state] )
        obs[t] = draw_state( B[state] )
    return obs

def draw_state(distr):
    """ helper function for random_sequence to get the state to a given probability

    Parameters
    ----------
    distr : array with probabilities where state are the indices

    Returns
    -------
    state : integer
            which randomly chosen with given distribution
    """
    x = np.random.random()
    D = len(distr)
    for state in range(D):
        if x < distr[state]:
            return state
        else:
            x -= distr[state]
    return state

__author__ = 'noe'
