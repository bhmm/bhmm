__author__ = 'noe'

import numpy as np

class OutputModelGaussian:
    """
    HMM output probability model using 1D-Gaussians

    """

    def __init__(self):
        pass

    def set_hmm(self, hmm_model):
        """
        Sets the output model. In this case this is a set of means and variances

        Parameters
        ----------
        means : ndarray (N)
            the mean values
        sigmas : ndarray (N)
            the variances

        """
        self.hmm_model = hmm_model
        self.means = np.zeros((hmm_model.nstates))
        self.sigmas = np.zeros((hmm_model.nstates))
        for i in range(hmm_model.nstates):
            self.means[i] = hmm_model.states[i]['mu']
            self.sigmas[i] = hmm_model.states[i]['sigma']

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
        mu = self.means[i]
        sigma = self.sigmas[i]
        C = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        Pobs = C * np.exp(-0.5 * ((o-mu)/sigma)**2)
        return Pobs

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
        C = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigmas)
        Pobs = C * np.exp(-0.5 * ((o-self.means)/self.sigmas)**2)
        return Pobs

    def p_obs(self, obs):
        """
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        oobs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states

        """
        T = len(obs)
        N = self.hmm_model.nstates
        res = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            res[t,:] = self.p_o(obs[t])
        return res

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
        K = len(observations)

        # fit means
        self.means  = np.zeros((N))
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

        # update model
        for i in range(N):
            self.hmm_model.states[i]['mu'] = self.means[i]
            self.hmm_model.states[i]['sigma'] = self.sigmas[i]


    def generate(self, observations):
        K = len(observations)
        X = np.empty(K, dtype = object)
        for k in range(K):
            T = len(observations[k])
            X[k] = np.zeros((T))
            for t in range(T):
                s = observations[k][t]
                X[k][t] = np.random.normal(self.means[s], self.sigmas[s])
        return X

