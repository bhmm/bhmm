"""
Hidden Markov model representation.

"""

import numpy as np

class HMM(object):
    """
    Hidden Markov model (HMM).

    This class is used to represent an HMM. This could be a maximum-likelihood HMM or a sampled HMM from a Bayesian posterior.

    Examples
    --------

    >>> nstates = 2
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> states = [ {'model' : 'gaussian', 'mu' : -1, 'sigma' : 1}, {'model' : 'gaussian', 'mu' : +1, 'sigma' : 1} ]
    >>> model = HMM(nstates, Tij, states)

    """
    def __init__(self, nstates, Tij, states, dtype=np.float64):
        """
        Parameters
        ----------
        nstates : int
            The number of discrete output states.
        Tij : np.array with shape (nstates, nstates), optional, default=None
            Row-stochastic transition matrix among states.
            If `None`, the identity matrix will be used.
        states : list of dict
            `states[i]` is a dict of parameters for state `i`, with Gaussian output parameters `mu` (mean) and `sigma` (standard deviation).

        """
        # TODO: Perform sanity checks on data consistency.

        self.nstates = nstates
        self.Tij = Tij # TODO: Rename to 'transition matrix'?
        self.Pi = self._compute_stationary_probabilities(self.Tij) # TODO: Rename to 'stationary_probabilities'?
        self.states = states # TODO: Rename to 'state_emission_parameters'?
        self.hidden_state_trajectories = None

        return

    @property
    def logPi(self):
        return np.log(self.Pi)

    @property
    def logTij(self):
        return np.log(self.Tij)

    def count_matrix(self, dtype=np.float64):
        """Compute the transition count matrix from hidden state trajectory.

        Parameters
        ----------
        dtype : numpy.dtype, optional, default=numpy.int32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        C : numpy.array with shape (nstates,nstates)
            C[i,j] is the number of transitions observed from state i to state j

        Raises
        ------
        RuntimeError
            A RuntimeError is raised if the HMM model does not yet have a hidden state trajectory associated with it.

        Examples
        --------

        """

        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        C = np.zeros((self.nstates,self.nstates), dtype=type)
        for S in self.hidden_state_trajectories:
            for t in range(len(S)-1):
                C[S[t],S[t+1]] += 1
        return C

    def emission_probability(self, state, observation):
        """Compute the emission probability of an observation from a given state.

        Parameters
        ----------
        state : int
            The state index for which the emission probability is to be computed.

        Returns
        -------
        Pobs : float
            The probability (or probability density, if continuous) of the observation.

        TODO
        ----
        * Vectorize

        Examples
        --------

        Compute the probability of observing an emission of 0 from state 0.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model(nstates=3)
        >>> state_index = 0
        >>> observation = 0.0
        >>> Pobs = model.emission_probability(state_index, observation)

        """
        observation_model = self.states[state]['model']
        if observation_model == 'gaussian':
            mu = self.states[state]['mu']
            sigma = self.states[state]['sigma']
            C = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
            Pobs = C * np.exp(-0.5 * ((observation-mu)/sigma)**2)
        else:
            raise Exception('Observation model "%s" unknown.' % observation_model)

        return Pobs

    def log_emission_probability(self, state, observation):
        """Compute the log emission probability of an observation from a given state.

        Parameters
        ----------
        state : int
            The state index for which the emission probability is to be computed.

        Returns
        -------
        log_Pobs : float
            The log probability (or probability density, if continuous) of the observation.

        TODO
        ----
        * Vectorize

        Examples
        --------

        Compute the log probability of observing an emission of 0 from state 0.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model(nstates=3)
        >>> state_index = 0
        >>> observation = 0.0
        >>> log_Pobs = model.log_emission_probability(state_index, observation)

        """
        observation_model = self.states[state]['model']
        if observation_model == 'gaussian':
            mu = self.states[state]['mu']
            sigma = self.states[state]['sigma']
            C = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
            log_Pobs = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((observation-mu)/sigma)**2
        else:
            raise Exception('Observation model "%s" unknown.' % observation_model)

        return log_Pobs

    def collect_observations_in_state(self, observations, state_index, dtype=np.float64):
        """Collect a vector of all observations belonging to a specified hidden state.

        Parameters
        ----------
        observations : list of numpy.array
            List of observed trajectories.
        state_index : int
            The index of the hidden state for which corresponding observations are to be retrieved.
        dtype : numpy.dtype, optional, default=numpy.float64
            The numpy dtype to use to store the collected observations.

        Returns
        -------
        collected_observations : numpy.array with shape (nsamples,)
            The collected vector of observations belonging to the specified hidden state.

        Raises
        ------
        RuntimeError
            A RuntimeError is raised if the HMM model does not yet have a hidden state trajectory associated with it.

        """
        if not self.hidden_state_trajectories:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        collected_observations = np.array([], dtype=dtype)
        for (s_t, o_t) in zip(self.hidden_state_trajectories, observations):
            indices = np.where(s_t == state_index)
            np.append(collected_observations, o_t[indices])

        return collected_observations

    def generate_synthetic_state_trajectory(self, length, initial_Pi=None, dtype=np.int32):
        """Generate a synthetic state trajectory.

        Parameters
        ----------
        length : int
            Length of synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.
        dtype : numpy.dtype, optional, default=numpy.int32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        states : np.array of shape (nstates,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,nstates).

        Examples
        --------

        Generate a synthetic state trajectory of a specified length.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> states = model.generate_synthetic_state_trajectory(length=100)

        """
        states = np.zeros([length], dtype=dtype)

        # Generate first state sample.
        if initial_Pi != None:
            states[0] = np.random.choice(range(self.nstates), size=1, p=initial_Pi)
        else:
            states[0] = np.random.choice(range(self.nstates), size=1, p=self.Pi)

        # Generate subsequent samples.
        for t in range(1,length):
            states[t] = np.random.choice(range(self.nstates), size=1, p=self.Tij[states[t-1],:])

        return states

    def generate_synthetic_observation(self, state):
        """Generate a synthetic observation from a given state.

        Parameters
        ----------
        state : int
            The index of the state from which the observable is to be sampled.

        Returns
        -------
        observation : float
            The observation from the given state.

        Examples
        --------

        Generate a synthetic observation from a single state.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> state_index = 0
        >>> observation = model.generate_synthetic_observation(state_index)

        """
        observation_model = self.states[state]['model']
        if observation_model == 'gaussian':
            observation = self.states[state]['sigma'] * np.random.randn() + self.states[state]['mu']
        else:
            raise Exception('Observation model "%s" unknown.' % observation_model)

        return observation

    def generate_synthetic_observation_trajectory(self, length, initial_Pi=None, dtype=np.float32):
        """Generate a synthetic realization of observables.

        Parameters
        ----------
        length : int
            Length of synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.
        dtype : numpy.dtype, optional, default=numpy.float32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        o_t : np.array of shape (nstates,) of dtype=np.float32
            The trajectory of observations.
        s_t : np.array of shape (nstates,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,nstates).

        Examples
        --------

        Generate a synthetic observation trajectory for an equilibrium realization.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100, initial_Pi=np.array([1,0,0]))

        """
        # First, generate synthetic state trajetory.
        s_t = self.generate_synthetic_state_trajectory(length, initial_Pi=initial_Pi)

        # Next, generate observations from these states.
        o_t = np.zeros([length], dtype=dtype)
        for t in range(length):
            o_t[t] = self.generate_synthetic_observation(s_t[t])

        return [o_t, s_t]

    def generate_synthetic_observation_trajectories(self, ntrajectories, length, initial_Pi=None, dtype=np.float32):
        """Generate a number of synthetic realization of observables from this model.

        Parameters
        ----------
        ntrajectories : int
            The number of trajectories to be generated.
        length : int
            Length of synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from equilibrium.
        dtype : numpy.dtype, optional, default=numpy.float32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        O : list of np.array of shape (nstates,) of dtype=np.float32
            The trajectories of observations
        S : list of np.array of shape (nstates,) of dtype=np.int32
            The trajectories of hidden states

        Examples
        --------

        Generate a number of synthetic trajectories.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> [O, S] = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
        >>> model = testsystems.three_state_model()
        >>> [O, S] = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100, initial_Pi=np.array([1,0,0]))


        """
        O = list() # observations
        S = list() # state trajectories
        for trajectory_index in range(ntrajectories):
            [o_t, s_t] = self.generate_synthetic_observation_trajectory(length=length, initial_Pi=initial_Pi, dtype=dtype)
            O.append(o_t)
            S.append(s_t)

        return [O, S]

    @classmethod
    def _compute_stationary_probabilities(cls, Tij, tol=1e-5, maxits=None, method='arpack'):
        """Compute the stationary probabilities for a given transition matrix.

        Parameters
        ----------
        Tij : numpy.array with shape (nstates, nstates)
            The row-stochastic transition matrix for which the stationary probabilities are to be computed.
        tol : float, optional, default=1e-5
            The absolute tolerance in total variation distance between probability vector iterates at which iterations are terminated.
        maxits : int, optional, default=None
            If not None, the maximum number of iterations to perform.
        method : str, optional, default='arpack'
            Method of stationary eigenvector computation: ['arpack', 'inverse-iteration']

        Returns
        -------
        Pi : numpy.array with shape (nstates, )
            The stationary probabilities corresponding to the row-stochastic transition matrix Tij.

        Notes
        -----
        This function uses the inverse iteration: http://en.wikipedia.org/wiki/Inverse_iteration

        Examples
        --------

        Compute stationary probabilities for a given transition matrix.

        >>> from bhmm import testsystems
        >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
        >>> Pi = HMM._compute_stationary_probabilities(Tij)

        """
        nstates = Tij.shape[0]

        if nstates == 2:
            # Use analytical method for 2x2 matrices.
            return np.array([Tij[1,0], Tij[0,1]], np.float64)

        # For larger matrices, solve numerically.
        if method == 'arpack':
            # Compute stationary probability using ARPACK.
            # TODO: Pass 'maxits' and 'tol' to ARPACK?
            from scipy.sparse.linalg import eigs
            from numpy.linalg import norm
            [eigenvalues, eigenvectors] = eigs(Tij.T, k=1, which='LR')
            eigenvectors = np.real(eigenvectors)
            Pi = eigenvectors[:,0] / eigenvectors[:,0].sum()
            return Pi
        elif method == 'inverse-iteration':
            T = np.array(Tij, dtype=np.float64) # Promote matrix to float64
            mu = 1.0 # eigenvalue corresponding to eigenvector to extract
            I = np.eye(nstates, dtype=np.float64) # identity matrix
            b_old = np.ones([nstates], dtype=np.float64) / float(nstates) # initial eigenvector guess

            # Perform inverse iteration.
            converged = False
            iteration = 1
            while not converged:
                # Update eigenvector guess
                b_new = np.dot(np.linalg.inv((T - mu*I).T), b_old)

                # Normalize to be a probability.
                b_new /= b_new.sum()

                # Compute total variation probability difference.
                delta = 0.5 * np.absolute(b_new - b_old).sum()

                # Check convergence criterion.
                if maxits:
                    converged = (iteration >= maxits)
                if tol:
                    converged = (delta < tol)

                # Proceed to next iteration.
                if not converged:
                    iteration += 1
                    b_old = b_new

            # Normalize vector to sum to unity.
            Pi = b_new / b_new.sum()
            return Pi
        else:
            raise Exception("method %s unknown." % method)

        return
