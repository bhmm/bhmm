"""
Hidden Markov model

"""

import numpy as np

class MLHMM(object):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    Examples
    --------

    >>> import testsystems
    >>> [model, observations] = testsystems.generate_synthetic_observations()
    >>> mlhmm = MLHMM(observations, model.nstates)
    >>> model = mlhmm.fit()

    """
    def __init__(self, observations, nstates, initial_model=None, verbose=False):
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

        """
        self.verbose = verbose

        # Store the number of states.
        self.nstates = nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        # Determine number of observation trajectories we have been given.
        self.ntrajectories = len(self.observations)

        if initial_model:
            # Use user-specified initial model, if provided.
            self.model = copy.deepcopy(initial_model)
        else:
            # Generate our own initial model.
            self.model = self._generateInitialModel()

        return

    def fit(self):
        """Fit a maximum-likelihood HMM model.

        Returns
        -------
        model : HMM
            The maximum likelihood HMM model.

        """
        # TODO: EM procedure.

        return self.model

    def _generateInitialModel(self):
        """Use a heuristic scheme to generate an initial model.

        TODO
        ----
        * Replace this with EM or MLHMM procedure from Matlab code.

        """
        nstates = self.nstates

        # Concatenate all observations.
        collected_observations = np.array([], dtype=dtype)
        for (o_t) in self.observations:
            np.append(collected_observations, o_t[indices])

        # Fit a Gaussian mixture model to obtain emission distributions and state stationary probabilities.
        from sklearn import mixture
        gmm = mixture.GMM(n_components=nstates)
        gmm.fit(collected_observations)
        states = list()
        for state_index in range(nstates):
            state = { 'model' : 'gaussian', 'mu' : gmm.means_[state_index,0], 'sigma' : np.sqrt(gmm.covars_[state_index,0,0]) }
            states.append(state)
        Pi = gmm.weights_

        # Compute fractional membership.
        Nij = np.zeros([nstates, nstates], np.float64)
        for trajectory_index in self.ntrajectories:
            o_t = self.observations[trajectory_index] # extract trajectory
            T = o_t.shape[0]
            #p_ti = self._computeStateProbabilities(o_t, states, Pi) # compute fractional memberships
            # Accumulate fractional transition counts from this trajectory.
            #for t in range(T):
            #    Nij[:,:] = Nij[:,:] + p_ti[t,:].T * p_ti[t+1,:]

        # DEBUG
        from bhmm import testsystems
        model = testsystems.generate_random_model(nstates=self.nstates)
        return model

