__author__ = 'noe'

import numpy as np
import copy
import kernel.python_obs as kp

class BaumWelchHMM:

    def __init__(self, output_model, observations, initial_model,
                 kernel = kp, dtype = np.float32,
                 accuracy=1e-3, maxit=1000):

        # Use user-specified initial model
        self.model = copy.deepcopy(initial_model)

        # Store the output model
        output_model.set_hmm(self.model)
        self.output_model = output_model

        # Store the number of states.
        self.nstates = initial_model.nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        # Determine number of observation trajectories we have been given.
        self.ntrajectories = len(self.observations)

        # Kernel for computing things
        self.kernel = kernel

        # dtype
        self.dtype = dtype

        # convergence options
        self.accuracy = accuracy
        self.maxit = maxit
        self.likelihoods = np.zeros((maxit))


    def _forward_backward(self, itraj):
        # get parameters
        A = self.model.Tij
        pi = self.model.Pi
        obs = self.observations[itraj]
        # compute output probability matrix
        pobs = self.output_model.p_obs(obs)
        # forward backward
        weight, alpha, scaling = self.kernel.forward(A, pobs, pi, self.dtype)
        beta   = self.kernel.backward(A, pobs, self.dtype)
        gamma  = self.kernel.state_probabilities(alpha, beta, self.dtype)
        # count matrix
        count_matrix = self.kernel.transition_counts(alpha, beta, A, pobs, self.dtype)
        return weight, gamma, count_matrix


    def _update_multiple(self, gammas, count_matrices):
        K = len(self.observations)
        N = self.nstates

        C = np.zeros((N,N))
        gamma0_sum = np.zeros((N))
        for k in range(K):
            # update state counts
            gamma0_sum += gammas[k][0]
            # update count matrix
            C += count_matrices[k]

        # compute new transition matrix
        A = C / np.sum(C,axis=1)[:,None]
        pi = gamma0_sum / np.sum(gamma0_sum)

        # update model
        self.model.Tij = copy.deepcopy(A)
        self.model.Pi  = copy.deepcopy(pi)



    def fit(self):
        """
        Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm

        Returns
        -------
        The hidden markov model

        """
        K = len(self.observations)#, len(A), len(B[0])
        gammas = np.empty(K, dtype=object)
        count_matrices = np.empty(K, dtype=object)

        it        = 0
        converged = False

        while (not converged):
            loglik = 0.0
            for k in range(K):
                w, gammas[k], count_matrices[k] = self._forward_backward(k)
                loglik += np.sum(w)

            # update T, pi
            self._update_multiple(gammas, count_matrices)

            # update output model
            self.output_model.fit(self.observations, gammas)

            self.likelihoods[it] = loglik

            if it > 0:
                if loglik - self.likelihoods[it-1] < self.accuracy:
                    converged = True

            it += 1

        return self.output_model.hmm_model