__author__ = 'noe'

import copy
import numpy as np
from math import log

import bhmm.output_models
from bhmm.output_models import OutputModel

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "Frank Noe"
__email__="frank DOT noe AT fu-berlin DOT de"

class DiscreteOutputModel(OutputModel):
    """
    HMM output probability model using discrete symbols. This is the "standard" HMM that is classically used in the
    literature

    """

    def __init__(self, B):
        """
        Create a 1D Gaussian output model.

        Parameters
        ----------
        B : ndarray((N,M),dtype=float)
            output probability matrix using N hidden states and M observable symbols.
            This matrix needs to be row-stochastic.

        Examples
        --------

        Create an observation model.

        >>> import numpy as np
        >>> B = np.array([[0.5,0.5],[0.1,0.9]])
        >>> output_model = DiscreteOutputModel(B)

        """
        self.B = np.array(B, dtype=np.float64)
        self.nstates,self.nsymbols = self.B.shape[0],self.B.shape[1]
        # test if row-stochastic
        assert np.allclose(np.sum(self.B, axis=1), np.ones(self.nstates)), 'B is not a stochastic matrix'
        # set output matrix
        self.B = B

    def __repr__(self):
        """
        >>> import numpy as np
        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))
        >>> print repr(output_model)
        DiscreteOutputModel(array([[ 0.5,  0.5],
               [ 0.1,  0.9]]))

        """

        return "DiscreteOutputModel(%s)" % repr(self.B)

    def __str__(self):
        """
        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))
        >>> print str(output_model)
        --------------------------------------------------------------------------------
        DiscreteOutputModel
        nstates: 2
        nsymbols: 2
        B[0] = [ 0.5  0.5]
        B[1] = [ 0.1  0.9]
        --------------------------------------------------------------------------------
        """

        output  = "--------------------------------------------------------------------------------\n"
        output += "DiscreteOutputModel\n"
        output += "nstates: %d\n" % self.nstates
        output += "nsymbols: %d\n" % self.nsymbols
        for i in range(self.nstates):
            output += "B["+str(i)+"] = %s\n" % str(self.B[i])
        output += "--------------------------------------------------------------------------------"
        return output

    # TODO: remove this code if we're sure we don't need it.
    # def p_o_i(self, o, i):
    #     """
    #     Returns the output probability for symbol o given hidden state i
    #
    #     Parameters
    #     ----------
    #     o : int
    #         the discrete symbol o (observation)
    #     i : int
    #         the hidden state index
    #
    #     Return
    #     ------
    #     p_o : float
    #         the probability that hidden state i generates symbol o
    #
    #     """
    #     # TODO: so far we don't use this method. Perhaps we don't need it.
    #     return self.B[i,o]
    #
    # def log_p_o_i(self, o, i):
    #     """
    #     Returns the logarithm of the output probability for symbol o given hidden state i
    #
    #     Parameters
    #     ----------
    #     o : int
    #         the discrete symbol o (observation)
    #     i : int
    #         the hidden state index
    #
    #     Return
    #     ------
    #     p_o : float
    #         the log probability that hidden state i generates symbol o
    #
    #     """
    #     # TODO: check if we need the log-probabilities
    #     return log(self.B[i,o])
    #
    #
    # def p_o(self, o):
    #     """
    #     Returns the output probability for symbol o from all hidden states
    #
    #     Parameters
    #     ----------
    #     o : int
    #         the discrete symbol o (observation)
    #
    #     Return
    #     ------
    #     p_o : ndarray (N)
    #         the probability that any of the N hidden states generates symbol o
    #
    #     """
    #     # TODO: so far we don't use this method. Perhaps we don't need it.
    #     return self.B[:,o]
    #
    # def log_p_o(self, o):
    #     """
    #     Returns the logarithm of the output probabilities for symbol o from all hidden states
    #
    #     Parameters
    #     ----------
    #     o : int
    #         the discrete symbol o (observation)
    #
    #     Return
    #     ------
    #     p_o : ndarray (N)
    #         the log probability that any of the N hidden states generates symbol o
    #
    #     """
    #     return np.log(self.B[:,o])

    def p_obs(self, obs, out=None, dtype=np.float32):
        """
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        obs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states

        """
        # much faster
        if (out is None):
            return self.B[:,obs].T
        else:
            if (obs.shape[0] == out.shape[0]):
                out[:,:] = self.B[:,obs].T
            elif (obs.shape[0] < out.shape[0]):
                out[:obs.shape[0],:] = self.B[:,obs].T
            else:
                raise ValueError('output array out is too small: '+str(out.shape[0])+' < '+str(obs.shape[0]))
            return out


    def fit(self, observations, weights):
        """
        Fits the output model given the observations and weights

        Parameters
        ----------

        observations : [ ndarray(T_k) ] with K elements
            A list of K observation trajectories, each having length T_k
        weights : [ ndarray(T_k,N) ] with K elements
            A list of K weight matrices, each having length T_k and containing the probability of any of the states in
            the given time step

        Examples
        --------

        Generate an observation model and samples from each state.

        >>> import numpy as np
        >>> ntrajectories = 3
        >>> nobs = 1000
        >>> B = np.array([[0.5,0.5],[0.1,0.9]])
        >>> output_model = DiscreteOutputModel(B)

        >>> from scipy import stats
        >>> nobs = 1000
        >>> obs = np.empty((nobs), dtype = object)
        >>> weights = np.empty((nobs), dtype = object)

        >>> gens = [stats.rv_discrete(values=(range(len(B[i])), B[i])) for i in range(B.shape[0])]
        >>> obs = [gens[i].rvs(size=nobs) for i in range(B.shape[0])]
        >>> weights = [np.zeros((nobs, B.shape[1])) for i in range(B.shape[0])]
        >>> for i in range(B.shape[0]): weights[i][:,i] = 1.0

        Update the observation model parameters my a maximum-likelihood fit.

        >>> output_model.fit(obs, weights)

        """
        # sizes
        N = self.B.shape[0]
        M = self.B.shape[1]
        K = len(observations)
        # initialize output probability matrix
        self.B  = np.zeros((N,M))
        for k in range(K):
            # update nominator
            obs = observations[k]
            for o in range(M):
                times = np.where(obs == o)[0]
                self.B[:,o] += np.sum(weights[k][times,:], axis=0)

        # normalize
        for o in range(M):
            sum = np.sum(self.B[:,o])
            if sum > 0:
                self.B[:,o] /= np.sum(self.B[:,o])

    def sample(self, observations):
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        Both the internal parameters and the attached HMM model are updated.

        Parameters
        ----------
        observations :  [ numpy.array with shape (N_k,) ] with nstates elements
            observations[k] is a set of observations sampled from state k

        Examples
        --------

        initialize output model

        >>> B = np.array([[0.5,0.5],[0.1,0.9]])
        >>> output_model = DiscreteOutputModel(B)

        sample given observation

        >>> obs = [[0,0,0,1,1,1],[1,1,1,1,1,1]]
        >>> output_model.sample(obs)

        """
        from numpy.random import dirichlet
        # total number of observation symbols
        M = self.B.shape[1]
        count_full = np.zeros((M), dtype = int)
        for i in range(len(observations)):
            # count symbols found in data
            count = np.bincount(observations[i])
            # blow up to full symbol space (if symbols are missing in this observation)
            count_full[:count.shape[0]] = count[:]
            # sample dirichlet distribution
            self.B[i,:] = dirichlet(count_full + 1)

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

        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))

        Generate sample from each state.

        >>> observation = output_model.generate_observation_from_state(0)

        """
        # generate random generator (note that this is inefficient - better use one of the next functions
        import scipy.stats
        gen = scipy.stats.rv_discrete(values=(range(len(self.B[state_index])), self.B[state_index]))
        gen.rvs(size=1)

    def generate_observations_from_state(self, state_index, nobs):
        """
        Generate synthetic observation data from a given state.

        Parameters
        ----------
        state_index : int
            Index of the state from which observations are to be generated.
        nobs : int
            The number of observations to generate.

        Returns
        -------
        observations : numpy.array of shape(nobs,) with type dtype
            A sample of `nobs` observations from the specified state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))

        Generate sample from each state.

        >>> observations = [ output_model.generate_observations_from_state(state_index, nobs=100) for state_index in range(output_model.nstates) ]

        """
        import scipy.stats
        gen = scipy.stats.rv_discrete(values=(range(self.nsymbols), self.B[state_index]))
        gen.rvs(size=nobs)

    def generate_observation_trajectory(self, s_t, dtype=np.int32):
        """
        Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        s_t : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]
        dtype : numpy.dtype, optional, default=numpy.int32
            The datatype to return the resulting observations in.

        Examples
        --------

        Generate an observation model and synthetic state trajectory.

        >>> nobs = 1000
        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """

        # Determine number of samples to generate.
        T = s_t.shape[0]
        # generate random generators
        import scipy.stats
        gens = [scipy.stats.rv_discrete(values=(range(len(self.B[state_index])), self.B[state_index])) for state_index in range(self.B.shape[0])]
        o_t = np.zeros([T], dtype=dtype)
        for t in range(T):
            s = s_t[t]
            o_t[t] = gens[s].rvs(size=1)
        return o_t



