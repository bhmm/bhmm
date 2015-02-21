__author__ = 'noe'

import numpy as np
import copy
import kernel.python as kp
from multiprocessing import Queue, JoinableQueue, Process, cpu_count
import bhmm.msm.linalg


class BaumWelchHMM:
    """
    Baum-Welch maximum likelihood method of estimating a Hidden Markov Model

    References
    ----------
    [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical estimation for probabilistic
        functions of a Markov process and to a model for ecology," Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.
    """

    def __init__(self, observations, initial_model,
                 kernel = kp, dtype = np.float32,
                 accuracy=1e-3, maxit=1000):

        # Use user-specified initial model
        self.model = copy.deepcopy(initial_model)

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
        """
        Estimation step: Runs the forward-back algorithm on trajectory with index itraj

        Parameters
        ----------
        itraj : int
            index of the observation trajectory to process

        Results
        -------
        logprob : float
            The probability to observe the observation sequence given the HMM parameters
        gamma : ndarray(T,N, dtype=float)
            state probabilities for each t
        count_matrix : ndarray(N,N, dtype=float)
            the Baum-Welch transition count matrix from the hidden state trajectory

        """
        # get parameters
        A = self.model.Tij
        pi = self.model.Pi
        obs = self.observations[itraj]
        # compute output probability matrix
        pobs = self.model.output_model.p_obs(obs)
        # forward backward
        logprob, alpha, scaling = self.kernel.forward(A, pobs, pi, self.dtype)
        beta   = self.kernel.backward(A, pobs, self.dtype)
        gamma  = self.kernel.state_probabilities(alpha, beta, self.dtype)
        # count matrix
        count_matrix = self.kernel.transition_counts(alpha, beta, A, pobs, self.dtype)
        return logprob, gamma, count_matrix


    def _update_model(self, gammas, count_matrices):
        """
        Maximization step: Updates the HMM model given the hidden state assignment and count matrices

        Parameters
        ----------
        gamma : [ ndarray(T,N, dtype=float) ]
            list of state probabilities for each trajectory
        count_matrix : [ ndarray(N,N, dtype=float) ]
            list of the Baum-Welch transition count matrices for each hidden state trajectory

        """
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
        if self.model.reversible:
            T = bhmm.msm.linalg.transition_matrix_MLE_reversible(C)
        else:
            T = bhmm.msm.linalg.transition_matrix_MLE_nonreversible(C)
        # stationary or init distribution
        if self.model.stationary:
            pi = bhmm.msm.linalg.stationary_distribution(T)
        else:
            pi = gamma0_sum / np.sum(gamma0_sum)

        # update model
        self.model.Tij = copy.deepcopy(T)
        self.model.Pi  = copy.deepcopy(pi)

        # update output model
        # TODO: need to parallelize model fitting. Otherwise we can't gain much speed!
        self.model.output_model.fit(self.observations, gammas)


    def viterbi_paths(self):
        """
        Computes the viterbi paths using the current HMM model

        """
        # get parameters
        K = len(self.observations)
        A = self.model.Tij
        pi = self.model.Pi

        # compute viterbi path for each trajectory
        hidden = np.empty((K), dtype=object)
        for itraj in range(K):
            obs = self.observations[itraj]
            # compute output probability matrix
            pobs = self.model.output_model.p_obs(obs)
            # hidden path
            hidden[itraj] = self.kernel.viterbi(A, pobs, pi)

        # done
        return hidden


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
                ll, gammas[k], count_matrices[k] = self._forward_backward(k)
                loglik += ll

            self._update_model(gammas, count_matrices)
            print it, "ll = ", loglik
            #print self.model.output_model
            #print "---------------------"

            self.likelihoods[it] = loglik

            if it > 0:
                if loglik - self.likelihoods[it-1] < self.accuracy:
                    converged = True

            it += 1

        return self.model

    ###################
    # MULTIPROCESSING

    def _forward_backward_worker(self, work_queue, done_queue):
        try:
            for k in iter(work_queue.get, 'STOP'):
                (weight, gamma, count_matrix) = self._forward_backward(k)
                done_queue.put((k, weight, gamma, count_matrix))
        except Exception, e:
            done_queue.put(e.message)
        return True


    def fit_parallel(self):
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

        num_threads = min(cpu_count(), K)
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        while (not converged):
            print "it", it
            loglik = 0.0

            # fill work queue
            for k in range(K):
                work_queue.put(k)

            # start processes
            for w in xrange(num_threads):
                p = Process(target=self._forward_backward_worker, args=(work_queue, done_queue))
                p.start()
                processes.append(p)
                work_queue.put('STOP')

            # end processes
            for p in processes:
                p.join()

            # done signal
            done_queue.put('STOP')

            # get results
            for (k, ll, gamma, count_matrix) in iter(done_queue.get, 'STOP'):
                loglik += ll
                gammas[k] = gamma
                count_matrices[k] = count_matrix

            # update T, pi
            self._update_model(gammas, count_matrices)

            self.likelihoods[it] = loglik

            if it > 0:
                if loglik - self.likelihoods[it-1] < self.accuracy:
                    converged = True

            it += 1

        return self.model
