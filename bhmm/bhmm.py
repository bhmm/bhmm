"""
Bayesian hidden Markov models.

"""

import copy

class BHMM(object):
    """Bayesian hidden Markov model sampler.

    Examples
    --------

    First, create some synthetic test data.

    >>> from bhmm import testsystems
    >>> model = testsystems.three_state_model()
    >>> data = testsystems.generate_synthetic_data(model, ntraces=10, length=100)

    Initialize a new BHMM model.

    >>> bhmm = BHMM(data)

    Sample from the posterior.

    >>> models = bhmm.sample(nsamples=10)

    """
    def __init__(observations, initial_model=None):
        """Initialize a Bayesian hidden Markov model sampler.

        Parameters
        ----------
        observations : list of 1d numpy arrays
            `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`

        """
        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

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

    def _updateStateTrajectories(self, model):
        """Sample a new set of state trajectories from the conditional distribution P(S | T, O)

        Parameters
        ----------
        model : bhmm.HMM
            The model for which a new set of hidden state trajectories is to be sampled. Will be modified.

        """
        pass

    def _updateEmissionProbabilities(self, model):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O)

        Parameters
        ----------
        model : bhmm.HMM
            The model for which a new set of emission probabilities is to be sampled. Will be modified.

        """
        pass

    def _updateTransitionMatrix(self, model):
        pass
