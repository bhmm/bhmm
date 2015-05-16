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
import bhmm
import bhmm.hidden as hidden
from bhmm.util.logger import logger
from bhmm.util import config

class MaximumLikelihoodEstimator(object):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    Examples
    --------

    >>> import bhmm
    >>> bhmm.config.verbose = False
    >>>
    >>> from bhmm import testsystems
    >>> [model, O, S] = testsystems.generate_synthetic_observations(ntrajectories=5, length=1000)
    >>> mlhmm = MaximumLikelihoodEstimator(O, model.nstates)
    >>> model = mlhmm.fit()

    References
    ----------
    [1] L. E. Baum and J. A. Egon, "An inequality with applications to statistical estimation for probabilistic
        functions of a Markov process and to a model for ecology," Bull. Amer. Meteorol. Soc., vol. 73, pp. 360-363, 1967.

    """
    def __init__(self, observations, nstates, initial_model=None, type='gaussian',
                 reversible=True, stationary=True, p=None, accuracy=1e-3, maxit=1000):
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
        type : str, optional, default=None
            Output model type from [None, 'gaussian', 'discrete'].
        reversible : bool, optional, default=True
            If True, a prior that enforces reversible transition matrices (detailed balance) is used;
            otherwise, a standard  non-reversible prior is used.
        stationary : bool, optional, default=True
            If True, the initial distribution of hidden states is self-consistently computed as the stationary
            distribution of the transition matrix. If False, it will be estimated from the starting states.
        p : ndarray (nstates), optional, default=None
            Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
            estimated with the constraint that they have p as their stationary distribution. If given and
            stationary=False, p is the fixed initial distribution of hidden states.
        accuracy : float
            convergence threshold for EM iteration. When two the likelihood does not increase by more than accuracy, the
            iteration is stopped successfully.
        maxit : int
            stopping criterion for EM iteration. When so many iterations are performanced without reaching the requested
            accuracy, the iteration is stopped without convergence (a warning is given)

        """
        # Store a copy of the observations.
        self._observations = copy.deepcopy(observations)
        self._nobs = len(observations)
        self._Ts = [len(o) for o in observations]
        self._maxT = np.max(self._Ts)

        # Store the number of states.
        self._nstates = nstates


        if initial_model is not None:
            # Use user-specified initial model, if provided.
            self._hmm = copy.deepcopy(initial_model)
            # consistency checks
            if self._hmm.is_stationary != stationary:
                logger().warn('Requested stationary='+str(stationary)+' but initial model is stationary='+
                              str(self._hmm.is_stationary)+'. Using stationary='+str(self._hmm.is_stationary))
            if self._hmm.is_reversible != reversible:
                logger().warn('Requested reversible='+str(reversible)+' but initial model is reversible='+
                              str(self._hmm.is_reversible)+'. Using reversible='+str(self._hmm.is_reversible))
            # setting parameters
            self._reversible = self._hmm.is_reversible
            self._stationary = self._hmm.is_stationary
        else:
            # Generate our own initial model.
            self._hmm = bhmm.init_hmm(observations, nstates, type=type)
            # setting parameters
            self._reversible = reversible
            self._stationary = stationary

        # stationary and initial distribution
        self._fixed_stationary_distribution = None
        self._fixed_initial_distribution = None
        if p is not None:
            if stationary:
                self._fixed_stationary_distribution = np.array(p)
            else:
                self._fixed_initial_distribution = np.array(p)

        # pre-construct hidden variables
        self._alpha = np.zeros((self._maxT,self._nstates), config.dtype, order='C')
        self._beta = np.zeros((self._maxT,self._nstates), config.dtype, order='C')
        self._pobs = np.zeros((self._maxT,self._nstates), config.dtype, order='C')
        self._gammas = [np.zeros((len(self._observations[i]),self._nstates), config.dtype, order='C') for i in range(self._nobs)]
        self._Cs = [np.zeros((self._nstates,self._nstates), config.dtype, order='C') for i in range(self._nobs)]

        # convergence options
        self._accuracy = accuracy
        self._maxit = maxit
        self._likelihoods = None

        # Kernel for computing things
        hidden.set_implementation(config.kernel)
        self._hmm.output_model.set_implementation(config.kernel)

    @property
    def observations(self):
        r""" Observation trajectories """
        return self._observations

    @property
    def nobservations(self):
        r""" Number of observation trajectories """
        return self._nobs

    @property
    def observation_lengths(self):
        r""" Lengths of observation trajectories """
        return self._Ts

    @property
    def is_reversible(self):
        r""" Whether the transition matrix is estimated with detailed balance constraints """
        return self._reversible

    @property
    def nstates(self):
        r""" Number of hidden states """
        return self._nstates

    @property
    def accuracy(self):
        r""" Convergence threshold for EM iteration """
        return self._accuracy

    @property
    def maxit(self):
        r""" Maximum number of iterations """
        return self._maxit

    @property
    def likelihood(self):
        r""" Estimated HMM likelihood """
        return self._likelihoods[-1]

    @property
    def likelihoods(self):
        r""" Sequence of likelihoods generated from the iteration """
        return self._likelihoods

    @property
    def hidden_state_probabilities(self):
        r""" Probabilities of hidden states at every trajectory and time point """
        return self._gammas

    @property
    def hmm(self):
        r""" The estimated HMM """
        return self._hmm

    @property
    def output_model(self):
        r""" The HMM output model """
        return self._hmm.output_model

    @property
    def transition_matrix(self):
        r""" Hidden transition matrix """
        return self._hmm.Tij

    @property
    def initial_probability(self):
        r""" Initial probability """
        return self._hmm.Pi

    @property
    def stationary_probability(self):
        r""" Stationary probability, if the model is stationary """
        assert self._stationary, 'Estimator is not stationary'
        return self._hmm.Pi

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
        A = self._hmm.transition_matrix
        pi = self._hmm.stationary_distribution
        obs = self._observations[itraj]
        T = len(obs)
        # compute output probability matrix
        self._hmm.output_model.p_obs(obs, out=self._pobs)
        # forward variables
        logprob = hidden.forward(A, self._pobs, pi, T = T, alpha_out=self._alpha)[0]
        # backward variables
        hidden.backward(A, self._pobs, T = T, beta_out=self._beta)
        # gamma
        hidden.state_probabilities(self._alpha, self._beta, gamma_out = self._gammas[itraj])
        # count matrix
        hidden.transition_counts(self._alpha, self._beta, A, self._pobs, out = self._Cs[itraj])
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
        K = len(self._observations)
        N = self._nstates

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
        T = estimate_P(C, reversible=self._hmm.is_reversible, fixed_statdist=self._fixed_stationary_distribution)
        # stationary or init distribution
        if self._hmm.is_stationary:
            if self._fixed_stationary_distribution is None:
                pi = stationary_distribution(C,T)
            else:
                pi = self._fixed_stationary_distribution
        else:
            if self._fixed_initial_distribution is None:
                pi = gamma0_sum / np.sum(gamma0_sum)
            else:
                pi = self._fixed_initial_distribution

        # update model
        self._hmm.update(T, pi)

        logger().info("T: \n"+str(T))
        logger().info("pi: \n"+str(pi))

        # update output model
        # TODO: need to parallelize model fitting. Otherwise we can't gain much speed!
        self._hmm.output_model._estimate_output_model(self._observations, gammas)

    def compute_viterbi_paths(self):
        """
        Computes the viterbi paths using the current HMM model

        """
        # get parameters
        K = len(self._observations)
        A = self._hmm.transition_matrix
        pi = self._hmm.stationary_distribution

        # compute viterbi path for each trajectory
        paths = np.empty((K), dtype=object)
        for itraj in range(K):
            obs = self._observations[itraj]
            # compute output probability matrix
            pobs = self._hmm.output_model.p_obs(obs)
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

        """
        logger().info("=================================================================")
        logger().info("Running Baum-Welch:")
        logger().info("  input observations: "+str(self.nobservations)+" of lengths "+str(self.observation_lengths))
        logger().info("  initial HMM guess:"+str(self._hmm))

        initial_time = time.time()

        it = 0
        self._likelihoods = np.zeros((self.maxit))
        loglik = 0.0
        converged = False

        while (not converged and it < self.maxit):
            loglik = 0.0
            for k in range(self._nobs):
                loglik += self._forward_backward(k)

            self._update_model(self._gammas, self._Cs)
            logger().info(str(it)+" ll = "+str(loglik))
            #print self.model.output_model
            #print "---------------------"

            self._likelihoods[it] = loglik

            if it > 0:
                if loglik - self._likelihoods[it-1] < self._accuracy:
                    #print "CONVERGED! Likelihood change = ",(loglik - self.likelihoods[it-1])
                    converged = True

            it += 1

        # truncate likelihood history
        self._likelihoods = self._likelihoods[:it]
        # set final likelihood
        self._hmm.likelihood = loglik

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("maximum likelihood HMM:"+str(self._hmm))
        logger().info("Elapsed time for Baum-Welch solution: %.3f s" % elapsed_time)
        logger().info("Computing Viterbi path:")

        initial_time = time.time()

        # Compute hidden state trajectories using the Viterbi algorithm.
        self._hmm.hidden_state_trajectories = self.compute_viterbi_paths()

        final_time = time.time()
        elapsed_time = final_time - initial_time

        logger().info("Elapsed time for Viterbi path computation: %.3f s" % elapsed_time)
        logger().info("=================================================================")

        return self._hmm


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
