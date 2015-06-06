"""
Bayesian hidden Markov models.

"""

import numpy as np
import copy
import time
#from scipy.misc import logsumexp
import bhmm.hidden as hidden
from bhmm.msm.tmatrix_disconnected import sample_P
from bhmm.util.logger import logger
from bhmm.util import config

#from bhmm.msm.transition_matrix_sampling_rev import TransitionMatrixSamplerRev

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "LGPL"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class BayesianHMMSampler(object):
    """Bayesian hidden Markov model sampler.

    Examples
    --------

    First, create some synthetic test data.

    >>> import bhmm
    >>> bhmm.config.verbose = False
    >>> nstates = 3
    >>> model = bhmm.testsystems.dalton_model(nstates)
    >>> [observations, hidden_states] = model.generate_synthetic_observation_trajectories(ntrajectories=5, length=1000)

    Initialize a new BHMM model.

    >>> from bhmm import BHMM
    >>> bhmm_sampler = BHMM(observations, nstates)

    Sample from the posterior.

    >>> models = bhmm_sampler.sample(nsamples=10)

    """
    def __init__(self, observations, nstates, initial_model=None,
                 reversible=True, transition_matrix_sampling_steps=1000, transition_matrix_prior=None,
                 type='gaussian'):
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
        transition_matrix_sampling_steps : int, optional, default=1000
            number of transition matrix sampling steps per BHMM cycle
        transition_matrix_prior : str or ndarray(n,n)
            prior count matrix to be used for transition matrix sampling, or a keyword specifying the prior mode
            |  None (default),  -1 prior is used that ensures consistency between mean and MLE. Can lead to sampling
                disconnected matrices in the low-data regime. If you have disconnectivity problems, consider
                using 'init-connect'
            |  'init-connect',  prior count matrix ensuring the same connectivity as in the initial model. 1 count
                is added to all diagonals. All off-diagonals share one prior count distributed proportional to
                the row of the initial transition matrix.
        output_model_type : str, optional, default='gaussian'
            Output model type.  ['gaussian', 'discrete']

        """
        # Sanity checks.
        if len(observations) == 0:
            raise Exception("No observations were provided.")

        # Store options.
        self.reversible = reversible

        # Store the number of states.
        self.nstates = nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)
        self.nobs = len(observations)
        self.Ts = [len(o) for o in observations]
        self.maxT = np.max(self.Ts)

        # initial model
        if initial_model:
            # Use user-specified initial model, if provided.
            self.model = copy.deepcopy(initial_model)
        else:
            # Generate our own initial model.
            self.model = self._generateInitialModel(type)

        # prior counts
        if transition_matrix_prior is None:
            self.prior = np.zeros((self.nstates, self.nstates))
        elif isinstance(transition_matrix_prior, np.ndarray):
            if np.array_equal(transition_matrix_prior.shape, (self.nstates, self.nstates)):
                self.prior = np.array(transition_matrix_prior)
        elif transition_matrix_prior == 'init-connect':
            Pinit = self.model.transition_matrix
            self.prior = Pinit - np.diag(Pinit)  # add off-diagonals from initial T-matrix
            self.prior /= self.prior.sum(axis=1)[:, None]  # scale off-diagonals to row sum 1
            self.prior += np.eye(nstates)  # add diagonal 1.
        else:
            raise ValueError('transition matrix prior mode undefined: '+str(transition_matrix_prior))

        # sampling options
        self.transition_matrix_sampling_steps = transition_matrix_sampling_steps

        # implementation options
        hidden.set_implementation(config.kernel)
        self.model.output_model.set_implementation(config.kernel)

        # pre-construct hidden variables
        self.alpha = np.zeros((self.maxT,self.nstates), config.dtype, order='C')
        self.pobs = np.zeros((self.maxT,self.nstates), config.dtype, order='C')

        return

    def sample(self, nsamples, nburn=0, nthin=1, save_hidden_state_trajectory=False):
        """Sample from the BHMM posterior.

        Parameters
        ----------
        nsamples : int
            The number of samples to generate.
        nburn : int, optional, default=0
            The number of samples to discard to burn-in, following which `nsamples` will be generated.
        nthin : int, optional, default=1
            The number of Gibbs sampling updates used to generate each returned sample.
        save_hidden_state_trajectory : bool, optional, default=False
            If True, the hidden state trajectory for each sample will be saved as well.

        Returns
        -------
        models : list of bhmm.HMM
            The sampled HMM models from the Bayesian posterior.

        Examples
        --------

        >>> from bhmm import testsystems
        >>> [model, observations, states, sampled_model] = testsystems.generate_random_bhmm(ntrajectories=5, length=1000)
        >>> nburn = 5 # run the sampler a bit before recording samples
        >>> nsamples = 10 # generate 10 samples
        >>> nthin = 2 # discard one sample in between each recorded sample
        >>> samples = sampled_model.sample(nsamples, nburn=nburn, nthin=nthin)

        """

        # Run burn-in.
        for iteration in range(nburn):
            logger().info("Burn-in   %8d / %8d" % (iteration, nburn))
            self._update()

        # Collect data.
        models = list()
        for iteration in range(nsamples):
            logger().info("Iteration %8d / %8d" % (iteration, nsamples))
            # Run a number of Gibbs sampling updates to generate each sample.
            for thin in range(nthin):
                self._update()
            # Save a copy of the current model.
            model_copy = copy.deepcopy(self.model)
            #print "Sampled: \n",repr(model_copy)
            if not save_hidden_state_trajectory:
                model_copy.hidden_state_trajectory = None
            models.append(model_copy)

        # Return the list of models saved.
        return models

    def _update(self):
        """Update the current model using one round of Gibbs sampling.

        """
        initial_time = time.time()

        self._updateHiddenStateTrajectories()
        self._updateEmissionProbabilities()
        self._updateTransitionMatrix()

        final_time = time.time()
        elapsed_time = final_time - initial_time
        logger().info("BHMM update iteration took %.3f s" % elapsed_time)

    def _updateHiddenStateTrajectories(self):
        """Sample a new set of state trajectories from the conditional distribution P(S | T, E, O)

        """
        self.model.hidden_state_trajectories = list()
        for trajectory_index in range(self.nobs):
            hidden_state_trajectory = self._sampleHiddenStateTrajectory(self.observations[trajectory_index])
            self.model.hidden_state_trajectories.append(hidden_state_trajectory)
        return

    def _sampleHiddenStateTrajectory(self, obs, dtype=np.int32):
        """Sample a hidden state trajectory from the conditional distribution P(s | T, E, o)

        Parameters
        ----------
        o_t : numpy.array with dimensions (T,)
            observation[n] is the nth observation
        dtype : numpy.dtype, optional, default=numpy.int32
            The dtype to to use for returned state trajectory.

        Returns
        -------
        s_t : numpy.array with dimensions (T,) of type `dtype`
            Hidden state trajectory, with s_t[t] the hidden state corresponding to observation o_t[t]

        Examples
        --------
        >>> import bhmm
        >>> [model, observations, states, sampled_model] = bhmm.testsystems.generate_random_bhmm(ntrajectories=5, length=1000)
        >>> o_t = observations[0]
        >>> s_t = sampled_model._sampleHiddenStateTrajectory(o_t)

        """

        # Determine observation trajectory length
        T = obs.shape[0]

        # Convenience access.
        A = self.model.transition_matrix
        pi = self.model.initial_distribution

        # compute output probability matrix
        self.model.output_model.p_obs(obs, out=self.pobs)
        # forward variables
        logprob = hidden.forward(A, self.pobs, pi, T = T, alpha_out=self.alpha)[0]
        # sample path
        S = hidden.sample_path(self.alpha, A, self.pobs, T = T)

        return S

    def _updateEmissionProbabilities(self):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O)

        """
        observations_by_state = [ self.model.collect_observations_in_state(self.observations, state) for state in range(self.model.nstates) ]
        self.model.output_model._sample_output_mode(observations_by_state)
        return

    def _updateTransitionMatrix(self):
        """
        Updates the hidden-state transition matrix

        """
        C = self.model.count_matrix()
        # apply prior
        C += self.prior
        # sample T-matrix
        Tij = sample_P(C, self.transition_matrix_sampling_steps, reversible=self.reversible)
        self.model.update(Tij)

    def _generateInitialModel(self, output_model_type):
        """Initialize using an MLHMM.

        """
        logger().info("Generating initial model for BHMM using MLHMM...")
        from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator
        mlhmm = MaximumLikelihoodEstimator(self.observations, self.nstates, reversible=self.reversible, type=output_model_type)
        model = mlhmm.fit()
        return model

