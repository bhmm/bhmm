__author__ = 'noe'

import copy
import numpy as np

class OutputModelDiscrete:
    """
    HMM output probability model using discrete symbols. This is the "standard" HMM that is classically used in the
    literature

    """

    def __init__(self):
        """

        :return:
        """
        pass

    def set_hmm(self, hmm_model):
        """
        Initializes the output model

        Parameters
        ----------
        hmm_model : hmm_class
            an instance of hmm_class

        """
        self.hmm_model = hmm_model
        B = [hmm_model.states[i]['pout'] for i in range(hmm_model.nstates)]
        self.B = np.vstack(B)

    def p_o_i(self, o, i):
        """
        Returns the output probability for symbol o given hidden state i

        Parameters
        ----------
        o : int
            the discrete symbol o (observation)
        i : int
            the hidden state index

        Return
        ------
        p_o : float
            the probability that hidden state i generates symbol o

        """
        # TODO: so far we don't use this method. Perhaps we don't need it.
        return self.B[i,o]

    def p_o(self, o):
        """
        Returns the output probability for symbol o from all hidden states

        Parameters
        ----------
        o : int
            the discrete symbol o (observation)

        Return
        ------
        p_o : ndarray (N)
            the probability that any of the N hidden states generates symbol o

        """
        return self.B[:,o]

    def p_obs(self, obs):
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
        # TODO: so far we don't use this method. Perhaps we don't need it.
        T = len(obs)
        N = self.hmm_model.nstates
        res = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            res[t,:] = self.B[:,obs[t]]
        return res

    # TODO: what about having a p_obs_i(self, i) that gives the observation probability for one state?
    # TODO: That could be sufficient, because it allows us to do efficient vector operations and is able to do state-based processing


    def fit(self, observations, weights):
        """
        Fits the output model given the observations and weights

        Parameters
        ----------
        observations : [ ndarray(T_k,d) ] with K elements
            A list of K observation trajectories, each having length T_k and d dimensions
        weights : [ ndarray(T_k,N) ] with K elements
            A list of K weight matrices, each having length T_k and containing the probability of any of the states in
            the given time step

        """
        # sizes
        N = self.hmm_model.nstates
        M = len(self.hmm_model.states[0]['pout'])
        K = len(observations)
        # initialize output probability matrix
        self.B  = np.zeros((N,M))
        for k in range(K):
            # update nominator
            obs = observations[k]
            for o in range(M):
                times = np.where(obs == o)[0]
                self.B[:,o] = np.sum(weights[k][times,:], axis=0)

        # normalize
        for o in range(M):
            self.B[:,o] /= np.sum(self.B[:,o])

        # update model
        for i in range(N):
            self.hmm_model.states[i]['pout'] = self.B[i]
