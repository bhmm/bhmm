__author__ = 'noe'

import unittest

import numpy as np

import bhmm.hidden as hidden

class TestHidden(unittest.TestCase):

    def setUp(self):
        A = np.array([[0.9, 0.1],
                      [0.1, 0.9]])
        pobs = np.array([[0.1, 0.9],
                         [0.1, 0.9],
                         [0.1, 0.9],
                         [0.1, 0.9],
                         [0.5, 0.5],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1],
                         [0.9, 0.1]])
        pi = np.array([0.5, 0.5])

        # PYTHON ALLOCATE IMPL
        hidden.set_implementation('python')
        (self.logprob_p, self.alpha_p, self.beta_p, self.gamma_p, self.statecount_p, self.C_p, self.vpath_p) = self.run_all(A, pobs, pi)
        # PYTHON PRE-ALLOCATE IMPL
        hidden.set_implementation('python')
        (self.logprob_p_mem, self.alpha_p_mem, self.beta_p_mem, self.gamma_p_mem, self.statecount_p_mem, self.C_p_mem, self.vpath_p_mem) = self.run_all(A, pobs, pi)

        # C ALLOCATE IMPL
        hidden.set_implementation('c')
        (self.logprob_c, self.alpha_c, self.beta_c, self.gamma_c, self.statecount_c, self.C_c, self.vpath_c) = self.run_all(A, pobs, pi)
        # C PRE-ALLOCATE IMPL
        hidden.set_implementation('c')
        (self.logprob_c_mem, self.alpha_c_mem, self.beta_c_mem, self.gamma_c_mem, self.statecount_c_mem, self.C_c_mem, self.vpath_c_mem) = self.run_all(A, pobs, pi)


    def run_all(self, A, pobs, pi):
        # forward
        logprob, alpha = hidden.forward(A, pobs, pi, dtype=np.float64)
        # backward
        beta = hidden.backward(A, pobs, dtype=np.float64)
        # gamma
        gamma = hidden.state_probabilities(alpha, beta)
        # state counts
        T = pobs.shape[0]
        statecount = hidden.state_counts(gamma, T)
        # transition counts
        C = hidden.transition_counts(alpha, beta, A, pobs, dtype=np.float64)
        # viterbi path
        vpath = hidden.viterbi(A, pobs, pi, dtype=np.float64)
        # return
        return (logprob, alpha, beta, gamma, statecount, C, vpath)

    def run_all_mem(self, A, pobs, pi):
        T = pobs.shape[0]
        N = A.shape[0]
        alpha = np.zeros( (T,N) )
        beta  = np.zeros( (T,N) )
        gamma = np.zeros( (T,N) )
        C     = np.zeros( (N,N) )
        logprob, alpha = hidden.forward(A, pobs, pi, alpha_out = alpha, dtype=np.float64)
        # backward
        hidden.backward(A, pobs, beta_out = beta, dtype=np.float64)
        # gamma
        hidden.state_probabilities(alpha_p, beta_p, gamma_out = gamma)
        # state counts
        statecount = hidden.state_counts(gamma, T)
        # transition counts
        hidden.transition_counts(alpha, beta, A, pobs, out=self.C, dtype=np.float64)
        # viterbi path
        vpath = hidden.viterbi(A, pobs, pi, dtype=np.float64)
        # return
        return (logprob, alpha, beta, gamma, statecount, C, vpath)

    def tearDown(self):
        pass

    def test_forward(self):
        # forward variables
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_c))
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_p_mem))
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_c_mem))

        self.assertTrue(np.allclose(self.alpha_p, self.alpha_c))
        self.assertTrue(np.allclose(self.alpha_p, self.alpha_p_mem))
        self.assertTrue(np.allclose(self.alpha_p, self.alpha_c_mem))

    def test_backward(self):
        # backward variables
        self.assertTrue(np.allclose(self.beta_p, self.beta_c))
        self.assertTrue(np.allclose(self.beta_p, self.beta_p_mem))
        self.assertTrue(np.allclose(self.beta_p, self.beta_c_mem))

    def test_gamma(self):
        # gammas / state probabilities
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_c))
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_p_mem))
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_c_mem))

    def test_counts(self):
        # state counts
        self.assertTrue(np.allclose(self.statecount_c, self.statecount_p))

        # transition counts
        self.assertTrue(np.allclose(self.C_p, self.C_c))
        self.assertTrue(np.allclose(self.C_p, self.C_p_mem))
        self.assertTrue(np.allclose(self.C_p, self.C_c_mem))

    def test_viterbi(self):
        # viterbi
        self.assertTrue(np.allclose(self.vpath_c, self.vpath_p))


if __name__=="__main__":
    unittest.main()
