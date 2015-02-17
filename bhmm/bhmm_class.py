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
        observations : list of 1d numpy arrays
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

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        # Determine number of observation trajectories we have been given.
        self.ntraces = len(self.observations)

        if initial_model:
            # Use user-specified initial model.
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
        """Sample a new set of state trajectories from the conditional distribution P(S | T, O)

        """
        pass

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

        """
        from bhmm import testsystems
        model = testsystems.generate_random_model(nstates=3)
        return model
