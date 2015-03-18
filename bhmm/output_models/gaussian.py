__author__ = 'noe'

import numpy as np

import impl_c.gaussian as gc
from bhmm.output_models import OutputModel


__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"



class GaussianOutputModel(OutputModel):
    """
    HMM output probability model using 1D-Gaussians

    """


    def __init__(self, nstates, means=None, sigmas=None):
        """
        Create a 1D Gaussian output model.

        Parameters
        ----------
        nstates : int
            The number of output states.
        means : array_like of shape (nstates,), optional, default=None
            If specified, initialize the Gaussian means to these values.
        sigmas : array_like of shape (nstates,), optional, default=None
            If specified, initialize the Gaussian variances to these values.

        Examples
        --------

        Create an observation model.

        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])

        """
        self.nstates = nstates

        dtype = np.float64 # type for internal storage

        if means is not None:
            self.means = np.array(means, dtype=dtype)
            if self.means.shape != (nstates,): raise Exception('means must have shape (%d,); instead got %s' % (nstates, str(self.means.shape)))
        else:
            self.means = np.zeros([nstates], dtype=dtype)

        if sigmas is not None:
            self.sigmas = np.array(sigmas, dtype=dtype)
            if self.sigmas.shape != (nstates,): raise Exception('sigmas must have shape (%d,); instead got %s' % (nstates, str(self.sigmas.shape)))
        else:
            self.sigmas = np.zeros([nstates], dtype=dtype)

        return


    def __repr__(self):
        """
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
        >>> print repr(output_model)
        GaussianOutputModel(3, means=array([-1.,  0.,  1.]), sigmas=array([ 0.5,  1. ,  2. ]))

        """

        return "GaussianOutputModel(%d, means=%s, sigmas=%s)" % (self.nstates, repr(self.means), repr(self.sigmas))

    def __str__(self):
        """
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
        >>> print str(output_model)
        --------------------------------------------------------------------------------
        GaussianOutputModel
        nstates: 3
        means: [-1.  0.  1.]
        sigmas: [ 0.5  1.   2. ]
        --------------------------------------------------------------------------------
        """

        output  = "--------------------------------------------------------------------------------\n"
        output += "GaussianOutputModel\n"
        output += "nstates: %d\n" % self.nstates
        output += "means: %s\n" % str(self.means)
        output += "sigmas: %s\n" % str(self.sigmas)
        output += "--------------------------------------------------------------------------------"
        return output

    # TODO: remove code when we're sure we don't need it
    # TODO: when cleaning up, save the functionality of the last function (log_p_o) and integrate into and overwrite of log_pobs.
    # def p_o_i(self, o, i):
    #     """
    #     Returns the output probability for symbol o given hidden state i
    #
    #     Parameters
    #     ----------
    #     o : float or array_like
    #         observation or observations for which probability is to be computed
    #     i : int
    #         the hidden state index
    #
    #     Return
    #     ------
    #     p_o_i : float
    #         the probability that hidden state i generates symbol o
    #
    #     Examples
    #     --------
    #
    #     Compute the output probability of a single observation from a given hidden state.
    #
    #     Create an observation model.
    #
    #     >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
    #
    #     Compute the output probability of a single observation from a single state.
    #
    #     >>> observation = 0
    #     >>> state_index = 0
    #     >>> p_o_i = output_model.p_o_i(observation, state_index)
    #
    #     Compute the output probability of a vector of observations from a single state.
    #
    #     >>> observations = np.random.randn(100)
    #     >>> state_index = 0
    #     >>> p_o_i = output_model.p_o_i(observations, state_index)
    #
    #     """
    #     C = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigmas[i])
    #     Pobs = C * np.exp(-0.5 * ((o - self.means[i]) / self.sigmas[i])**2)
    #     return Pobs
    #
    # def log_p_o_i(self, o, i):
    #     """
    #     Returns the log output probability for symbol o given hidden state i
    #
    #     Parameters
    #     ----------
    #     o : float or array_like
    #         observation or observations for which probability is to be computed
    #     i : int
    #         the hidden state index
    #
    #     Return
    #     ------
    #     log_p_o_i : float
    #         the probability that hidden state i generates symbol o
    #
    #     Examples
    #     --------
    #
    #     Compute the log output probability of a single observation from a given hidden state.
    #
    #     Create an observation model.
    #
    #     >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
    #
    #     Compute the output probability of a single observation from a single state.
    #
    #     >>> observation = 0
    #     >>> state_index = 0
    #     >>> log_p_o_i = output_model.log_p_o_i(observation, state_index)
    #
    #     Compute the output probability of a vector of observations from a single state.
    #
    #     >>> observations = np.random.randn(100)
    #     >>> state_index = 0
    #     >>> p_o_i = output_model.p_o_i(observations, state_index)
    #
    #     """
    #     log_C = - 0.5 * np.log(2.0 * np.pi) - np.log(self.sigmas[i])
    #     log_Pobs = log_C - 0.5 * ((o - self.means[i]) / self.sigmas[i])**2
    #     return log_Pobs
    #
    #
    # def p_o(self, o):
    #     """
    #     Returns the output probability for symbol o from all hidden states
    #
    #     Parameters
    #     ----------
    #     o : float
    #         A single observation.
    #
    #     Return
    #     ------
    #     p_o : ndarray (N)
    #         p_o[i] is the probability density of the observation o from state i emission distribution
    #
    #     Examples
    #     --------
    #
    #     Create an observation model.
    #
    #     >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
    #
    #     Compute the output probability of a single observation from all hidden states.
    #
    #     >>> observation = 0
    #     >>> p_o = output_model.p_o(observation)
    #
    #     """
    #     if self.__impl__ == self.__IMPL_C__:
    #         return gc.p_o(o, self.means, self.sigmas, out=None, dtype=type(o))
    #     elif self.__impl__ == self.__IMPL_PYTHON__:
    #         C = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigmas)
    #         Pobs = C * np.exp(-0.5 * ((o-self.means)/self.sigmas)**2)
    #         return Pobs
    #     else:
    #         raise RuntimeError('Implementation '+str(self.__impl__)+' not available')
    #
    #
    # def log_p_o(self, o):
    #     """
    #     Returns the log output probability for symbol o from all hidden states
    #
    #     Parameters
    #     ----------
    #     o : float
    #         A single observation.
    #
    #     Return
    #     ------
    #     log_p_o : ndarray (N)
    #         log_p_o[i] is the log probability density of the observation o from state i emission distribution
    #
    #     Examples
    #     --------
    #
    #     Create an observation model.
    #
    #     >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
    #
    #     Compute the output probability of a single observation from all hidden states.
    #
    #     >>> observation = 0
    #     >>> log_p_o = output_model.log_p_o(observation)
    #
    #     """
    #     log_C = - 0.5 * np.log(2.0 * np.pi) - np.log(self.sigmas)
    #     log_Pobs = log_C - 0.5 * ((o-self.means)/self.sigmas)**2
    #     return log_Pobs


    def p_obs(self, obs, out=None, dtype=np.float32):
        """
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        oobs : ndarray((T), dtype=int)
            a discrete trajectory of length T
        dtype : numpy.dtype, optional, default=numpy.float32
            The datatype to return the resulting observations in.

        Return
        ------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states

        Examples
        --------

        Generate an observation model and synthetic observation trajectory.

        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])
        >>> o_t = output_model.generate_observation_trajectory(s_t)

        Compute output probabilities for entire trajectory and all hidden states.

        >>> p_o = output_model.p_obs(o_t)

        """
        if self.__impl__ == self.__IMPL_C__:
            return gc.p_obs(obs, self.means, self.sigmas, out=out, dtype=dtype)
        elif self.__impl__ == self.__IMPL_PYTHON__:
            T = len(obs)
            if out is None:
                res = np.zeros((T, self.nstates), dtype=dtype)
            else:
                res = out
            for t in range(T):
                res[t,:] = self.p_o(obs[t])
            return res
        else:
            raise RuntimeError('Implementation '+str(self.__impl__)+' not available')


    def fit(self, observations, weights):
        """
        Fits the output model given the observations and weights

        Parameters
        ----------
        observations : [ ndarray(T_k,) ] with K elements
            A list of K observation trajectories, each having length T_k and d dimensions
        weights : [ ndarray(T_k,nstates) ] with K elements
            A list of K weight matrices, each having length T_k
            weights[k][t,n] is the weight assignment from observations[k][t] to state index n

        Examples
        --------

        Generate an observation model and samples from each state.

        >>> ntrajectories = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> observations = [ np.random.randn(nobs) for trajectory_index in range(ntrajectories) ] # random observations
        >>> weights = [ np.random.dirichlet([2, 3, 4], size=nobs) for trajectory_index in range(ntrajectories) ] # random weights

        Update the observation model parameters my a maximum-likelihood fit.

        >>> output_model.fit(observations, weights)

        """
        # sizes
        N = self.nstates
        K = len(observations)

        # fit means
        self.means = np.zeros((N))
        w_sum = np.zeros((N))
        for k in range(K):
            # update nominator
            for i in range(N):
                self.means[i] += np.dot(weights[k][:,i],observations[k])
            # update denominator
            w_sum += np.sum(weights[k], axis=0)
        # normalize
        self.means /= w_sum

        # fit variances
        self.sigmas  = np.zeros((N))
        w_sum = np.zeros((N))
        for k in range(K):
            # update nominator
            for i in range(N):
                Y = (observations[k]-self.means[i])**2
                self.sigmas[i] += np.dot(weights[k][:,i],Y)
            # update denominator
            w_sum += np.sum(weights[k], axis=0)
        # normalize
        self.sigmas /= w_sum
        self.sigmas = np.sqrt(self.sigmas)


    def sample(self, observations):
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        Both the internal parameters and the attached HMM model are updated.

        Parameters
        ----------
        observations :  [ numpy.array with shape (N_k,) ] with `nstates` elements
            observations[k] is a set of observations sampled from state `k`

        Examples
        --------

        Generate synthetic observations.

        >>> nstates = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=nstates, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
        >>> observations = [ output_model.generate_observations_from_state(state_index, nobs) for state_index in range(nstates) ]
        >>> weights = [ np.zeros([nobs,nstates], np.float32).T for state_index in range(nstates) ]

        Update output parameters by sampling.

        >>> output_model.sample(observations)

        """
        for state_index in range(self.nstates):
            # Update state emission distribution parameters.

            observations_in_state = observations[state_index]
            # Determine number of samples in this state.
            nsamples_in_state = len(observations_in_state)

            # Skip update if no observations.
            if nsamples_in_state == 0:
                print 'Warning: State %d has no obsevations.' % state_index
                continue

            # Sample new mu.
            self.means[state_index] = np.random.randn()*self.sigmas[state_index]/np.sqrt(nsamples_in_state) + np.mean(observations_in_state)

            # Sample new sigma.
            # This scheme uses the improper Jeffreys prior on sigma^2, P(mu, sigma^2) \propto 1/sigma
            chisquared = np.random.chisquare(nsamples_in_state-1)
            sigmahat2 = np.mean((observations_in_state - self.means[state_index])**2)
            self.sigmas[state_index] = np.sqrt(sigmahat2) / np.sqrt(chisquared / nsamples_in_state)

        return


    def generate_observation_from_state(self, state_index):
        """
        Generate a single synthetic observation data from a given state.

        Parameters
        ----------
        state_index : int
            Index of the state from which observations are to be generated.

        Returns
        -------
        observation : float
            A single observation from the given state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = GaussianOutputModel(nstates=2, means=[0, 1], sigmas=[1, 2])

        Generate sample from a state.

        >>> observation = output_model.generate_observation_from_state(0)

        """
        observation = self.sigmas[state_index] * np.random.randn() + self.means[state_index]
        return observation


    def generate_observations_from_state(self, state_index, nobs, dtype=np.float32):
        """
        Generate synthetic observation data from a given state.

        Parameters
        ----------
        state_index : int
            Index of the state from which observations are to be generated.
        nobs : int
            The number of observations to generate.
        dtype : numpy.dtype, optional, default=numpy.float32
            The datatype to return the resulting observations in.

        Returns
        -------
        observations : numpy.array of shape(nobs,) with type dtype
            A sample of `nobs` observations from the specified state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = GaussianOutputModel(nstates=2, means=[0, 1], sigmas=[1, 2])

        Generate samples from each state.

        >>> observations = [ output_model.generate_observations_from_state(state_index, nobs=100) for state_index in range(output_model.nstates) ]

        """
        observations = self.sigmas[state_index] * np.random.randn(nobs) + self.means[state_index]
        return observations


    def generate_observation_trajectory(self, s_t, dtype=np.float32):
        """
        Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        s_t : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t
        dtype : numpy.dtype, optional, default=numpy.float32
            The datatype to return the resulting observations in.

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]

        Examples
        --------

        Generate an observation model and synthetic state trajectory.

        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """

        # Determine number of samples to generate.
        T = s_t.shape[0]

        o_t = np.zeros([T], dtype=dtype)
        for t in range(T):
            s = s_t[t]
            o_t[t] = self.sigmas[s] * np.random.randn() + self.means[s]
        return o_t

