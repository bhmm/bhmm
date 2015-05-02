"""
Hidden Markov model

"""

__author__ = "Frank Noe and John D. Chodera"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["Frank Noe", "John D. Chodera"]
__license__ = "LGPL"
__maintainer__ = "Frank Noe"
__email__="frank DOT noe AT fu-berlin DOT de"

import time
import numpy as np
import copy

# TODO: reactivate multiprocessing
# from multiprocessing import Queue, Process, cpu_count

# BHMM imports
import bhmm.init as hmminit
import bhmm.hidden as hidden
from bhmm.util.logger import logger
from bhmm.util import config

class MaximumLikelihoodEstimator(object):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    Examples
    --------

    >>> from bhmm import testsystems
    >>> [model, O, S] = testsystems.generate_synthetic_observations()
    >>> mlhmm = MaximumLikelihoodEstimator(O, model.nstates)
    >>> model = mlhmm.fit()

    References
    ----------
    [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical estimation for probabilistic
        functions of a Markov process and to a model for ecology," Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.

    """
    def __init__(self, observations, nstates, initial_model=None, reversible=True, output_model_type='gaussian',
                 accuracy=1e-3, maxit=1000):
        """Initialize a Bayesian hidden Markov model sampler.

        Parameters
        ----------
        observations : list of numpy arrays representing temporal data
            `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
        nstates : int
            The number of states in the model.
        initial_model : HMM, optional, default=None
            If specified, the given initial model will be used to initialize the BHMM.
            Otherwise, a heuristic scheme is used to generate an initial guess.
        reversible : bool, optional, default=True
            If True, a prior that enforces reversible transition matrices (detailed balance) is used;
            otherwise, a standard  non-reversible prior is used.
        output_model_type : str, optional, default='gaussian'
            Output model type.  ['gaussian', 'discrete']
        dtype : type
            data type used for hidden state probabilities, transition probabilities and initial probilities
        accuracy : float
            convergence terminated for EM iteration. When two the likelihood does not increase by more than accuracy, the
            iteration is stopped successfully.
        maxit : int
            stopping criterion for EM iteration. When so many iterations are performanced without reaching the requested
            accuracy, the iteration is stopped without convergence (a warning is given)

        """
        # Store options.
        self.reversible = reversible

        # Store the number of states.
        self.nstates = nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)
        self.nobs = len(observations)
        self.Ts = [len(o) for o in observations]
        self.maxT = np.max(self.Ts)

        # Determine number of observation trajectories we have been given.
        self.ntrajectories = len(self.observations)

        if initial_model:
            # Use user-specified initial model, if provided.
            self.model = copy.deepcopy(initial_model)
        else:
            # Generate our own initial model.
            self.model = hmminit.generate_initial_model(observations, nstates, output_model_type)

        # Kernel for computing things
        hidden.set_implementation(config.kernel)
        self.model.output_model.set_implementation(config.kernel)

        # pre-construct hidden variables
        self.alpha = np.zeros((self.maxT,self.nstates), config.dtype, order='C')
        self.beta = np.zeros((self.maxT,self.nstates), config.dtype, order='C')
        self.pobs = np.zeros((self.maxT,self.nstates), config.dtype, order='C')
        self.gammas = [np.zeros((len(self.observations[i]),self.nstates), config.dtype, order='C') for i in range(self.nobs)]
        self.Cs = [np.zeros((self.nstates,self.nstates), config.dtype, order='C') for i in range(self.nobs)]

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
        T = len(obs)
        # compute output probability matrix
        self.model.output_model.p_obs(obs, out=self.pobs)
        # forward variables
        logprob = hidden.forward(A, self.pobs, pi, T = T, alpha_out=self.alpha)[0]
        # backward variables
        hidden.backward(A, self.pobs, T = T, beta_out=self.beta)
        # gamma
        hidden.state_probabilities(self.alpha, self.beta, gamma_out = self.gammas[itraj])
        # count matrix
        hidden.transition_counts(self.alpha, self.beta, A, self.pobs, out = self.Cs[itraj])
        # return results
        return logprob

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
            # print 'C['+str(k)+'] = ',count_matrices[k]
            C += count_matrices[k]


        logger().info("Count matrix = \n"+str(C))

        # compute new transition matrix
        from bhmm.msm.tmatrix_disconnected import estimate_P,stationary_distribution
        T = estimate_P(C, reversible=self.model.reversible)
        # stationary or init distribution
        if self.model.stationary:
            pi = stationary_distribution(C,T)
        else:
            pi = gamma0_sum / np.sum(gamma0_sum)

        # update model
        self.model.Tij = copy.deepcopy(T)
        self.model.Pi  = copy.deepcopy(pi)

        logger().info("T: \n"+str(T))
        logger().info("pi: \n"+str(pi))

        # update output model
        # TODO: need to parallelize model fitting. Otherwise we can't gain much speed!
        self.model.output_model._estimate_output_model(self.observations, gammas)

    @property
    def hidden_state_probabilities(self):
        return self.gammas

    @property
    def output_model(self):
        return self.model.output_model

    @property
    def transition_matrix(self):
        return self.model.Tij

    @property
    def initial_probability(self):
        return self.model.Pi

    @property
    def stationary_probability(self):
        return self.model.Pi

    @property
    def is_reversible(self):
        return self.model.reversible

    @property
    def is_stationary(self):
        return self.model.stationary

    def compute_viterbi_paths(self):
        """
        Computes the viterbi paths using the current HMM model

        """
        # get parameters
        K = len(self.observations)
        A = self.model.Tij
        pi = self.model.Pi

        # compute viterbi path for each trajectory
        paths = np.empty((K), dtype=object)
        for itraj in range(K):
            obs = self.observations[itraj]
            # compute output probability matrix
            pobs = self.model.output_model.p_obs(obs)
            # hidden path
            paths[itraj] = hidden.viterbi(A, pobs, pi)

        # done
        return paths

    def fit(self):
        """
        Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm

        Returns
        -------
        model : HMM
            The maximum likelihood HMM model.

        Examples
        --------

        >>> from bhmm import testsystems
        >>> [model, O, S] = testsystems.generate_synthetic_observations()
        >>> mlhmm = MaximumLikelihoodEstimator(O, model.nstates)
        >>> model = mlhmm.fit()

        """
        logger().info("================================================================================")
        logger().info("Running Baum-Welch:")
        logger().info("  input observations:\n"+str(self.observations))
        logger().info("  initial HMM guess:\n"+str(self.model))

        initial_time = time.time()

        it        = 0
        loglik = 0.0
        converged = False

        while (not converged):
            loglik = 0.0
            for k in range(self.nobs):
                loglik += self._forward_backward(k)

            self._update_model(self.gammas, self.Cs)
            logger().info(str(it)+" ll = "+str(loglik))
            #print self.model.output_model
            #print "---------------------"

            self.likelihoods[it] = loglik

            if it > 0:
                if loglik - self.likelihoods[it-1] < self.accuracy:
                    #print "CONVERGED! Likelihood change = ",(loglik - self.likelihoods[it-1])
                    converged = True

            it += 1

        # truncate likelihood history
        self.likelihoods = self.likelihoods[:it]
        # set final likelihood
        self.model.likelihood = loglik

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("maximum likelihood HMM:"+str(self.model))
        logger().info("Elapsed time for Baum-Welch solution: %.3f s" % elapsed_time)
        logger().info("\nComputing Viterbi path:")

        initial_time = time.time()

        # Compute hidden state trajectories using the Viterbi algorithm.
        self.hidden_state_trajectories = self.compute_viterbi_paths()

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("Elapsed time for Viterbi path computation: %.3f s" % elapsed_time)
        logger().info("================================================================================")

        return self.model


    # TODO: reactive multiprocessing
    # ###################
    # # MULTIPROCESSING
    #
    # def _forward_backward_worker(self, work_queue, done_queue):
    #     try:
    #         for k in iter(work_queue.get, 'STOP'):
    #             (weight, gamma, count_matrix) = self._forward_backward(k)
    #             done_queue.put((k, weight, gamma, count_matrix))
    #     except Exception, e:
    #         done_queue.put(e.message)
    #     return True
    #
    #
    # def fit_parallel(self):
    #     """
    #     Maximum-likelihood estimation of the HMM using the Baum-Welch algorithm
    #
    #     Returns
    #     -------
    #     The hidden markov model
    #
    #     """
    #     K = len(self.observations)#, len(A), len(B[0])
    #     gammas = np.empty(K, dtype=object)
    #     count_matrices = np.empty(K, dtype=object)
    #
    #     it        = 0
    #     converged = False
    #
    #     num_threads = min(cpu_count(), K)
    #     work_queue = Queue()
    #     done_queue = Queue()
    #     processes = []
    #
    #     while (not converged):
    #         print "it", it
    #         loglik = 0.0
    #
    #         # fill work queue
    #         for k in range(K):
    #             work_queue.put(k)
    #
    #         # start processes
    #         for w in xrange(num_threads):
    #             p = Process(target=self._forward_backward_worker, args=(work_queue, done_queue))
    #             p.start()
    #             processes.append(p)
    #             work_queue.put('STOP')
    #
    #         # end processes
    #         for p in processes:
    #             p.join()
    #
    #         # done signal
    #         done_queue.put('STOP')
    #
    #         # get results
    #         for (k, ll, gamma, count_matrix) in iter(done_queue.get, 'STOP'):
    #             loglik += ll
    #             gammas[k] = gamma
    #             count_matrices[k] = count_matrix
    #
    #         # update T, pi
    #         self._update_model(gammas, count_matrices)
    #
    #         self.likelihoods[it] = loglik
    #
    #         if it > 0:
    #             if loglik - self.likelihoods[it-1] < self.accuracy:
    #                 converged = True
    #
    #         it += 1
    #
    #     return self.model
