"""
Bayesian hidden Markov models.

"""

import numpy as np
import copy
import time
#from scipy.misc import logsumexp
import bhmm.hidden as hidden
from msm.tmatrix_disconnected import sample_P
from util.logger import logger


#from bhmm.msm.transition_matrix_sampling_rev import TransitionMatrixSamplerRev

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class BHMM(object):
    """Bayesian hidden Markov model sampler.

    Examples
    --------

    First, create some synthetic test data.

    >>> from bhmm import testsystems
    >>> nstates = 3
    >>> model = testsystems.dalton_model(nstates)
    >>> [observations, hidden_states] = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=10000)

    Initialize a new BHMM model.

    >>> from bhmm import BHMM
    >>> bhmm_sampler = BHMM(observations, nstates)

    Sample from the posterior.

    >>> models = bhmm_sampler.sample(nsamples=10)

    """
    def __init__(self, observations, nstates, initial_model=None,
                 reversible=True, transition_matrix_sampling_steps=1000,
                 output_model_type='gaussian',
                 dtype = np.float64, kernel = 'c'):
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
        output_model_type : str, optional, default='gaussian'
            Output model type.  ['gaussian', 'discrete']
        kernel: str, optional, default='python'
            Implementation kernel

        TODO
        ----
        Document choice of -1 prior for transition matrix samplng.

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
            self.model = self._generateInitialModel(output_model_type)

        # sampling options
        self.transition_matrix_sampling_steps = transition_matrix_sampling_steps

        # implementation options
        self.dtype = dtype
        self.kernel = kernel
        hidden.set_implementation(kernel)
        self.model.output_model.set_implementation(kernel)

        # pre-construct hidden variables
        self.alpha = np.zeros((self.maxT,self.nstates), dtype=dtype, order='C')
        self.pobs = np.zeros((self.maxT,self.nstates), dtype=dtype, order='C')

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
        >>> [model, observations, states, bhmm] = testsystems.generate_random_bhmm()
        >>> nburn = 5 # run the sampler a bit before recording samples
        >>> nsamples = 10 # generate 10 samples
        >>> nthin = 2 # discard one sample in between each recorded sample
        >>> samples = bhmm.sample(nsamples, nburn=nburn, nthin=nthin)

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
        logger.info("BHMM update iteration took %.3f s" % elapsed_time)

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
        >>> from bhmm import testsystems
        >>> [model, observations, states, bhmm] = testsystems.generate_random_bhmm()
        >>> o_t = observations[0]
        >>> s_t = bhmm._sampleHiddenStateTrajectory(o_t)

        """

        # Determine observation trajectory length
        T = obs.shape[0]

        # Convenience access.
        A = self.model.Tij
        pi = self.model.Pi

        # compute output probability matrix
        self.model.output_model.p_obs(obs, out=self.pobs, dtype=self.dtype)
        # forward variables
        logprob = hidden.forward(A, self.pobs, pi, T = T, alpha_out=self.alpha, dtype=self.dtype)[0]
        # sample path
        S = hidden.sample_path(self.alpha, A, self.pobs, T = T, dtype=self.dtype)

        return S

        # TODO: remove this when new impl. successfully tested.
        # model = self.model # current HMM model
        # nstates = model.nstates
        # logPi = model.logPi
        # logTij = model.logTij
        # #logPi = np.log(model.Pi)
        # #logTij = np.log(model.Tij)
        #
        # #
        # # Forward part.
        # #
        #
        # log_alpha_it = np.zeros([nstates, T], np.float64)
        #
        # for i in range(nstates):
        #     log_alpha_it[i,0] = logPi[i] + model.log_emission_probability(i, o_t[0])
        #
        # for t in range(1,T):
        #     for j in range(nstates):
        #         log_alpha_it[j,t] = logsumexp(log_alpha_it[:,t-1] + logTij[:,j]) + model.log_emission_probability(j, o_t[t])
        #
        # #
        # # Sample state trajectory in backwards part.
        # #
        #
        # s_t = np.zeros([T], dtype=dtype)
        #
        # # Sample final state.
        # log_p_i = log_alpha_it[:,T-1]
        # p_i = np.exp(log_p_i - logsumexp(log_alpha_it[:,T-1]))
        # s_t[T-1] = np.random.choice(range(nstates), size=1, p=p_i)
        #
        # # Work backwards from T-2 to 0.
        # for t in range(T-2, -1, -1):
        #     # Compute P(s_t = i | s_{t+1}..s_T).
        #     log_p_i = log_alpha_it[:,t] + logTij[:,s_t[t+1]]
        #     p_i = np.exp(log_p_i - logsumexp(log_p_i))
        #
        #     # Draw from this distribution.
        #     s_t[t] = np.random.choice(range(nstates), size=1, p=p_i)
        #
        # # Return trajectory
        # return s_t

    def _updateEmissionProbabilities(self):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O)

        """
        observations_by_state = [ self.model.collect_observations_in_state(self.observations, state) for state in range(self.model.nstates) ]
        self.model.output_model.sample(observations_by_state)
        return

    def _updateTransitionMatrix(self):
        """
        Updates the hidden-state transition matrix

        """
        C = self.model.count_matrix()
        self.model.Tij = sample_P(C, self.transition_matrix_sampling_steps, reversible=self.reversible)

    def _generateInitialModel(self, output_model_type):
        """Initialize using an MLHMM.

        """
        logger.info("Generating initial model for BHMM using MLHMM...")
        from bhmm import MLHMM
        mlhmm = MLHMM(self.observations, self.nstates, reversible=self.reversible, output_model_type=output_model_type)
        model = mlhmm.fit()
        return model

