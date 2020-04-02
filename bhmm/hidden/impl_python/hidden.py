
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Python implementation of Hidden Markov Model kernel functions

This module is considered to be the reference for checking correctness of other
kernels. All implementations are being kept very simple, straight forward and
closely related to Rabiners [1] paper.

.. [1] Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models and
   Selected Applications in Speech Recognition", Proceedings of the IEEE,
   vol. 77, issue 2
"""
import numpy as np


def forward(A, pobs, pi, T=None, alpha_out=None, dtype=np.float32):
    """Compute P( obs | A, B, pi ) and all forward coefficients.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    alpha_out : ndarray((T,N), dtype = float), optional, default = None
        container for the alpha result variables. If None, a new container will be created.
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = A.shape[0]
    # initialize output if necessary
    if alpha_out is None:
        alpha_out = np.zeros((T, N), dtype=dtype)
    elif T > alpha_out.shape[0]:
        raise ValueError('alpha_out must at least have length T in order to fit trajectory.')
    # log-likelihood
    logprob = 0.0

    # initial values
    # alpha_i(0) = pi_i * B_i,ob[0]
    np.multiply(pi, pobs[0, :], out=alpha_out[0])
    # scaling factor
    scale = np.sum(alpha_out[0, :])
    # scale
    alpha_out[0, :] /= scale
    logprob += np.log(scale)

    # induction
    for t in range(T-1):
        # alpha_j(t+1) = sum_i alpha_i(t) * A_i,j * B_j,ob(t+1)
        np.multiply(np.dot(alpha_out[t, :], A), pobs[t+1, :], out=alpha_out[t+1])
        # scaling factor
        scale = np.sum(alpha_out[t+1, :])
        # scale
        alpha_out[t+1, :] /= scale
        # update logprob
        logprob += np.log(scale)

    return logprob, alpha_out


def backward(A, pobs, T=None, beta_out=None, dtype=np.float32):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    beta_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the beta result variables. If None, a new container will be created.
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith backward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = A.shape[0]
    # initialize output if necessary
    if beta_out is None:
        beta_out = np.zeros((T, N), dtype=dtype)
    elif T > beta_out.shape[0]:
        raise ValueError('beta_out must at least have length T in order to fit trajectory.')

    # initialization
    beta_out[T-1, :] = 1.0
    # scaling factor
    scale = np.sum(beta_out[T-1, :])
    # scale
    beta_out[T-1, :] /= scale

    # induction
    for t in range(T-2, -1, -1):
        # beta_i(t) = sum_j A_i,j * beta_j(t+1) * B_j,ob(t+1)
        np.dot(A, beta_out[t+1, :] * pobs[t+1, :], out=beta_out[t, :])
        # scaling factor
        scale = np.sum(beta_out[t, :])
        # scale
        beta_out[t, :] /= scale
    return beta_out


def transition_counts(alpha, beta, A, pobs, T=None, out=None, dtype=np.float32):
    """ Sum for all t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int
        number of time steps
    out : ndarray((N,N), dtype = float), optional, default = None
        containter for the resulting count matrix. If None, a new matrix will be created.
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j in time [0,T)

    See Also
    --------
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = len(A)
    # output
    if out is None:
        out = np.zeros((N, N), dtype=dtype, order='C')
    else:
        out[:] = 0.0
    # compute transition counts
    xi = np.zeros((N, N), dtype=dtype, order='C')
    for t in range(T-1):
        # xi_i,j(t) = alpha_i(t) * A_i,j * B_j,ob(t+1) * beta_j(t+1)
        np.dot(alpha[t, :][:, None] * A, np.diag(pobs[t+1, :] * beta[t+1, :]), out=xi)
        # normalize to 1 for each time step
        xi /= np.sum(xi)
        # add to counts
        np.add(out, xi, out)
    # return
    return out


def viterbi(A, pobs, pi, dtype=np.float32):
    """ Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    T, N = pobs.shape[0], pobs.shape[1]
    # temporary viterbi state
    psi = np.zeros((T, N), dtype=int)
    # initialize
    v = pi * pobs[0, :]
    # rescale
    v /= v.sum()
    psi[0] = 0.0
    # iterate
    for t in range(1, T):
        vA = np.dot(np.diag(v), A)
        # propagate v
        v = pobs[t, :] * np.max(vA, axis=0)
        # rescale
        v /= v.sum()
        psi[t] = np.argmax(vA, axis=0)
    # iterate
    q = np.zeros(T, dtype=int)
    q[T-1] = np.argmax(v)
    for t in range(T-2, -1, -1):
        q[t] = psi[t+1, q[t+1]]
    # done
    return q


def sample_path(alpha, A, pobs, T=None, dtype=np.float32, seed=None):
    """
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    seed : int, optional, default=None
        Sets the internal random state of numpy if not None.
    """
    if seed is not None:
        np.random.seed(seed)
    N = pobs.shape[1]
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0] or T > alpha.shape[0]:
        raise ValueError('T must be at most the length of pobs and alpha.')

    # initialize path
    S = np.zeros(T, dtype=int)

    # Sample final state.
    psel = alpha[T-1, :]
    psel /= psel.sum()  # make sure it's normalized
    # Draw from this distribution.
    S[T-1] = np.random.choice(range(N), size=1, p=psel)

    # Work backwards from T-2 to 0.
    for t in range(T-2, -1, -1):
        # Compute P(s_t = i | s_{t+1}..s_T).
        psel = alpha[t, :] * A[:, S[t+1]]
        psel /= psel.sum()  # make sure it's normalized
        # Draw from this distribution.
        S[t] = np.random.choice(range(N), size=1, p=psel)

    return S
