"""
Bayesian hidden Markov models.

"""

import copy
from bhmm.msm.transition_matrix_sampling_rev import TransitionMatrixSamplerRev

class BHMM(object):
    """Bayesian hidden Markov model sampler.

    Examples
    --------

    First, create some synthetic test data.

    >>> from bhmm import testsystems
    >>> nstates = 3
    >>> model = testsystems.generate_random_model(nstates)
    >>> data = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100)

    Initialize a new BHMM model.

    >>> bhmm = BHMM(data, nstates)

    Sample from the posterior.

    >>> models = bhmm.sample(nsamples=10)

    """
    def __init__(self, observations, nstates, initial_model=None, verbose=False,
                 transition_matrix_sampling_steps = 1000):
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
        verbose : bool, optional, default=False
            Verbosity flag.
        transition_matrix_sampling_steps : int
            number of transition matrix sampling steps per BHMM cycle

        """
        self.verbose = verbose

        # Store the number of states.
        self.nstates = nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        # Determine number of observation trajectories we have been given.
        self.nobservations = len(self.observations)

        if initial_model:
            # Use user-specified initial model, if provided.
            self.model = copy.deepcopy(initial_model)
        else:
            # Generate our own initial model.
            self.model = self._generateInitialModel()

        self.transition_matrix_sampling_steps = transition_matrix_sampling_steps

        return

    def sample(self, nsamples, nburn=0, nthin=1):
        """Sample from the BHMM posterior.

        Parameters
        ----------
        nsamples : int
            The number of samples to generate.
        nburn : int, optional, default=0
            The number of samples to discard to burn-in, following which `nsamples` will be generated.
        nthin : int, optional, default=1
            The number of Gibbs sampling updates used to generate each returned sample.

        Returns
        -------
        models : list of bhmm.HMM
            The sampled HMM models from the Bayesian posterior.

        """

        # Run burn-in.
        for iteration in range(nburn):
            if self.verbose: print "Burn-in   %8d / %8d" % (iteration, nburn)
            self._update()

        # Collect data.
        models = list()
        for iteration in range(nsamples):
            if self.verbose: print "Iteration %8d / %8d" % (iteration, nsamples)
            # Run a number of Gibbs sampling updates to generate each sample.
            for thin in range(nthin):
                self._update()
            # Save a copy of the current model.
            models.append(copy.deepcopy(self.model))

        # Return the list of models saved.
        return models

    def _update(self):
        """Update the current model using one round of Gibbs sampling.

        """
        self._updateStateTrajectories(self.model)
        self._updateEmissionProbabilities(self.model)
        self._updateTransitionMatrix(self.model)


    def _updateStateTrajectories(self):
        """Sample a new set of state trajectories from the conditional distribution P(S | T, E, O)

        """
        self.model.hidden_state_trajectories = list()
        for trajectory_index in range(self.ntrajectories):
            hidden_state_trajectory = self._sampleHiddenStateTrajectory(self.observations[trajectory_index])
            self.model.hidden_state_trajectories.append(hidden_state_trajectory)
        return

    def _sampleHiddenStateTrajectory(self, o_t, dtype=np.int32):
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
        >>> [model, observations, bhmm] = generate_random_bhmm_model()
        >>> o_t = observations[0]
        >>> s_t = bhmm._sampleHiddenStateTrajectory(o_t)

        """

        # Determine observation trajectory length
        T = o_t.shape[0]

        # Convenience access.
        model = self.model # current HMM model
        nstates = model.nstates
        logPi = model.logPi
        logTij = model.logTij

        #
        # Forward part.
        #

        log_alpha_it = np.zeros([nstates, T], np.float64)

        # TODO: Vectorize in i.
        for i in range(nstates):
            log_alpha_it[i,0] = logPi[i] + model.log_emission_probability(i, o_t[0])

        # TODO: Vectorize in j.
        for t in range(1,T):
            for j in range(nstates):
                log_alpha_it[j,t] = np.logsumexp(log_alpha_it[:,t-1] + logTij[:,j]) + model.log_emission_probability(j, o_t[t])

        #
        # Sample state trajectory in backwards part.
        #

        s_t = np.zeros([T], dtype=dtype)

        # Sample final state.
        log_p_i = log_alpha_it[:,T-1]
        p_i = np.exp(log_p_i - np.logsumexp(log_alpha_it[:,T-1]))
        s_t[T-1] states[0] = np.random.choice(range(nstates), size=1, p=p_i)


        # Work backwards.
        for t in range(T-2, 0, -1):
            # Compute P(s_t = i | s_{t+1}..s_T).
            log_p_i = log_alpha_it[:,t] + logTij[:,s_t[t+1]]
            p_i = np.exp(log_p_i - np.logsumexp(log_p_i))

            # Draw from this distribution.
            s_t[t] = np.random.choice(range(nstates), size=1, p=p_i)

        # Return trajectory
        return s_t

    def _updateEmissionProbabilities(self):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O)

        """
        pass

    def _updateTransitionMatrix(self):
        """
        Updates the hidden-state transition matrix

        """
        C = self.model.count_matrix()
        sampler = TransitionMatrixSamplerRev(C)
        self.model.Tij = sampler.sample(self.transition_matrix_sampling_steps)

    def _generateInitialModel(self):
        """Use a heuristic scheme to generate an initial model.

        TODO
        ----
        * Replace this with EM or MLHMM procedure from Matlab code.

        """
        from bhmm import testsystems
        model = testsystems.generate_random_model(nstates=self.nstates)
        return model

