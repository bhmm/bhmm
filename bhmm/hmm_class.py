"""
Hidden Markov model representation.

"""

import numpy as np
#import msm.linalg as msmalg
import output_models

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class HMM(object):
    """
    Hidden Markov model (HMM).

    This class is used to represent an HMM. This could be a maximum-likelihood HMM or a sampled HMM from a Bayesian posterior.

    Examples
    --------

    >>> # Gaussian HMM
    >>> nstates = 2
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from output_models import GaussianOutputModel
    >>> output_model = GaussianOutputModel(nstates, means=[-1, +1], sigmas=[1, 1])
    >>> model = HMM(nstates, Tij, output_model)

    >>> # Discrete HMM
    >>> nstates = 2
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> from output_models import DiscreteOutputModel
    >>> output_model = DiscreteOutputModel([[0.5, 0.1, 0.4], [0.2, 0.3, 0.5]])
    >>> model = HMM(nstates, Tij, output_model)

    """
    def __init__(self, nstates, Tij, output_model,
                 Pi = None, stationary = True, # initial / stationary probability
                 reversible = True, # transition matrix reversible?
                 dtype=np.float64):
        """
        Parameters
        ----------
        nstates : int
            The number of discrete output states.
        Tij : np.array with shape (nstates, nstates), optional, default=None
            Row-stochastic transition matrix among states.
            If `None`, the identity matrix will be used.
        output_model : bhmm.OutputModel
            The output model for the states.
        reversible : bool, optional, default=True
            If True, will note that the transition matrix is reversible.

        """
        # TODO: Perform sanity checks on data consistency.
        # TODO: dtype seems not to be used

        self.nstates = nstates
        self.Tij = Tij # TODO: Rename to 'transition matrix'?
        self.reversible = reversible

        # initial / stationary distribution
        import pyemma.msm.analysis as msmana
        self.stationary = stationary
        if (stationary):
            self.Pi = msmana.stationary_distribution(self.Tij) # TODO: Rename to 'stationary_probabilities'?
        else:
            if Pi is None: # no initial distribution given, so use stationary distribution anyway
                self.Pi = msmana.stationary_distribution(self.Tij)
            else:
                self.Pi = Pi

        # output model
        self.output_model = output_model

        # hidden state trajectories are optional
        self.hidden_state_trajectories = None
        self.reversible = reversible

        return

    def __repr__(self):
        return "HMM(%d, %s, %s, Pi=%s, stationary=%s, reversible=%s, dtype=%s)" % (self.nstates, repr(self.Tij), repr(self.output_model), repr(self.Pi), repr(self.stationary), repr(self.reversible), repr(self.dtype))

    def __str__(self):
        output  = 'Hidden Markov model\n'
        output += '-------------------\n'
        output += 'nstates: %d\n' % self.nstates
        output += 'Tij:\n'
        output += str(self.Tij) + '\n'
        output += 'Pi:\n'
        output += str(self.Pi) + '\n'
        output += 'output model:\n'
        output += str(self.output_model)
        output += '\n'
        return output

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
        return self.output_model.p_o_i(observation, state)

    # def log_emission_probability(self, state, observation):
    #     """Compute the log emission probability of an observation from a given state.
    #
    #     Parameters
    #     ----------
    #     state : int
    #         The state index for which the emission probability is to be computed.
    #
    #     Returns
    #     -------
    #     log_Pobs : float
    #         The log probability (or probability density, if continuous) of the observation.
    #
    #     TODO
    #     ----
    #     * Vectorize
    #
    #     Examples
    #     --------
    #
    #     Compute the log probability of observing an emission of 0 from state 0.
    #
    #     >>> from bhmm import testsystems
    #     >>> model = testsystems.dalton_model(nstates=3)
    #     >>> state_index = 0
    #     >>> observation = 0.0
    #     >>> log_Pobs = model.log_emission_probability(state_index, observation)
    #
    #     """
    #     return self.output_model.log_p_o_i(observation, state)

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
            indices = np.where(s_t == state_index)[0]
            collected_observations = np.append(collected_observations, o_t[indices])

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
        >>> model = testsystems.dalton_model()
        >>> states = model.generate_synthetic_state_trajectory(length=100)

        """
        states = np.zeros([length], dtype=dtype)

        # Generate first state sample.
        if initial_Pi is not None:
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
        >>> model = testsystems.dalton_model()
        >>> observation = model.generate_synthetic_observation(0)

        """
        return self.output_model.generate_observation_from_state(state)

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
        >>> model = testsystems.dalton_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> [o_t, s_t] = model.generate_synthetic_observation_trajectory(length=100, initial_Pi=np.array([1,0,0]))

        """
        # First, generate synthetic state trajetory.
        s_t = self.generate_synthetic_state_trajectory(length, initial_Pi=initial_Pi)

        # Next, generate observations from these states.
        o_t = self.output_model.generate_observation_trajectory(s_t, dtype=dtype)

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
        >>> model = testsystems.dalton_model()
        >>> [O, S] = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100)

        Use an initial nonequilibrium distribution.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model(nstates=3)
        >>> [O, S] = model.generate_synthetic_observation_trajectories(ntrajectories=10, length=100, initial_Pi=np.array([1,0,0]))


        """
        O = list() # observations
        S = list() # state trajectories
        for trajectory_index in range(ntrajectories):
            [o_t, s_t] = self.generate_synthetic_observation_trajectory(length=length, initial_Pi=initial_Pi, dtype=dtype)
            O.append(o_t)
            S.append(s_t)

        return [O, S]

