__author__ = 'noe'

import unittest

import numpy as np

import bhmm.hidden as hidden

class TestBaumWelch(unittest.TestCase):

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
        T = pobs.shape[0]
        N = 2
        pi = np.array([0.5, 0.5])

        # in-memory
        self.alpha_c_mem = np.zeros( (T,N) )
        self.beta_c_mem = np.zeros( (T,N) )
        self.gamma_c_mem = np.zeros( (T,N) )
        self.C_c_mem = np.zeros( (N,N) )
        #
        self.alpha_p_mem = np.zeros( (T,N) )
        self.beta_p_mem = np.zeros( (T,N) )
        self.gamma_p_mem = np.zeros( (T,N) )
        self.C_p_mem = np.zeros( (N,N) )

        # PYTHON IMPL
        hidden.set_implementation('python')
        # forward
        self.logprob_p, self.alpha_p = hidden.forward(A, pobs, pi, dtype=np.float64)
        self.logprob_p_mem, self.alpha_p_mem = hidden.forward(A, pobs, pi, alpha_out = self.alpha_p_mem, dtype=np.float64)
        # backward
        self.beta_p = hidden.backward(A, pobs, dtype=np.float64)
        hidden.backward(A, pobs, beta_out = self.beta_p_mem, dtype=np.float64)
        # gamma
        self.gamma_p = hidden.state_probabilities(self.alpha_p, self.beta_p)
        hidden.state_probabilities(self.alpha_p, self.beta_p, gamma_out = self.gamma_p_mem)
        # state counts
        self.statecount_p = hidden.state_counts(self.gamma_p, T)
        # transition counts
        self.C_p = hidden.transition_counts(self.alpha_p, self.beta_p, A, pobs, dtype=np.float64)
        hidden.transition_counts(self.alpha_p, self.beta_p, A, pobs, out=self.C_p_mem, dtype=np.float64)
        # viterbi path
        self.vpath_p = hidden.viterbi(A, pobs, pi, dtype=np.float64)

        # C IMPL
        hidden.set_implementation('c')
        # forward
        self.logprob_c, self.alpha_c = hidden.forward(A, pobs, pi, dtype=np.float64)
        self.logprob_c_mem, self.alpha_c_mem = hidden.forward(A, pobs, pi, alpha_out = self.alpha_c_mem, dtype=np.float64)
        # backward
        self.beta_c = hidden.backward(A, pobs, dtype=np.float64)
        hidden.backward(A, pobs, beta_out = self.beta_c_mem, dtype=np.float64)
        # gamma
        self.gamma_c = hidden.state_probabilities(self.alpha_c, self.beta_c)
        hidden.state_probabilities(self.alpha_c, self.beta_c, gamma_out = self.gamma_c_mem)
        # state counts
        self.statecount_c = hidden.state_counts(self.gamma_c, T)
        # transition counts
        self.C_c = hidden.transition_counts(self.alpha_c, self.beta_c, A, pobs, dtype=np.float64)
        hidden.transition_counts(self.alpha_c, self.beta_c, A, pobs, out=self.C_c_mem, dtype=np.float64)
        # viterbi path
        self.vpath_c = hidden.viterbi(A, pobs, pi, dtype=np.float64)


    def tearDown(self):
        pass

    def test_connected_sets(self):
        print 'logprob'
        print self.logprob_c
        print self.logprob_p
        print
        print 'alpha'
        print self.alpha_c
        print self.alpha_p
        print
        print 'beta'
        print self.beta_c
        print self.beta_p
        print
        print 'gamma'
        print self.gamma_c
        print self.gamma_p
        print
        print 'state counts'
        print self.statecount_c
        print self.statecount_p
        print
        print 'transition counts'
        print self.C_c
        print self.C_p
        print
        print 'viterbi path'
        print self.vpath_c
        print self.vpath_p

        # forward variables
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_c))
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_p_mem))
        self.assertTrue(np.allclose(self.logprob_p, self.logprob_c_mem))

        self.assertTrue(np.allclose(self.alpha_p, self.alpha_c))
        self.assertTrue(np.allclose(self.alpha_p, self.alpha_p_mem))
        self.assertTrue(np.allclose(self.alpha_p, self.alpha_c_mem))

        # backward variables
        self.assertTrue(np.allclose(self.beta_p, self.beta_c))
        self.assertTrue(np.allclose(self.beta_p, self.beta_p_mem))
        self.assertTrue(np.allclose(self.beta_p, self.beta_c_mem))

        # gammas / state probabilities
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_c))
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_p_mem))
        self.assertTrue(np.allclose(self.gamma_p, self.gamma_c_mem))

        # state counts
        self.assertTrue(np.allclose(self.statecount_c, self.statecount_p))

        # transition counts
        self.assertTrue(np.allclose(self.C_p, self.C_c))
        self.assertTrue(np.allclose(self.C_p, self.C_p_mem))
        self.assertTrue(np.allclose(self.C_p, self.C_c_mem))

        # viterbi
        self.assertTrue(np.allclose(self.vpath_c, self.vpath_p))


if __name__=="__main__":
    unittest.main()
