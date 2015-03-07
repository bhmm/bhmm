__author__ = 'noe'

import unittest

import numpy as np
import bhmm.ml.kernel.python as bw_python
import bhmm.ml.kernel.c as bw_c

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
        T = pobs.shape[1]
        pi = np.array([0.5, 0.5])

        # forward
        self.logprob_python, self.alpha_python, self.scaling_python = bw_python.forward(A, pobs, pi, dtype=np.float64)
        self.logprob_c, self.alpha_c, self.scaling_c = bw_c.forward(A, pobs, pi, dtype=np.float64)

        # backward
        self.beta_python = bw_python.backward(A, pobs, dtype=np.float64)
        self.beta_c = bw_c.backward(A, pobs, dtype=np.float64)

        # backward
        self.gamma_python = bw_python.state_probabilities(self.alpha_python, self.beta_python, dtype=np.float64)
        self.gamma_c = bw_python.state_probabilities(self.alpha_c, self.beta_c, dtype=np.float64)

        # state counts
        self.statecount_python = bw_python.state_counts(self.gamma_python, T, dtype=np.float64)
        self.statecount_c = bw_c.state_counts(self.gamma_c, T, dtype=np.float64)

        # transition counts
        self.C_python = bw_python.transition_counts(self.alpha_python, self.beta_python, A, pobs, dtype=np.float64)
        self.C_c = bw_c.transition_counts(self.alpha_c, self.beta_c, A, pobs, dtype=np.float64)

        # viterbi path
        self.vpath_python = bw_python.viterbi(A, pobs, pi, dtype=np.float64)
        print "lala"
        print self.vpath_python

        self.vpath_c = bw_c.viterbi(A, pobs, pi, dtype=np.float64)


    def tearDown(self):
        pass

    def test_connected_sets(self):
        print 'logprob'
        print self.logprob_c
        print self.logprob_python
        print
        print 'alpha'
        print self.alpha_c
        print self.alpha_python
        print
        print 'scaling'
        print self.scaling_c
        print self.scaling_python
        print
        print 'beta'
        print self.beta_c
        print self.beta_python
        print
        print 'gamma'
        print self.gamma_c
        print self.gamma_python
        print
        print 'state counts'
        print self.statecount_c
        print self.statecount_python
        print
        print 'transition counts'
        print self.C_c
        print self.C_python
        print
        print 'viterbi path'
        #print self.vpath_c
        #print self.vpath_python

        # forward variables
        self.assertTrue(np.allclose(self.logprob_c, self.logprob_python))
        self.assertTrue(np.allclose(self.alpha_c, self.alpha_python))
        self.assertTrue(np.allclose(self.scaling_c, self.scaling_python))

        # backward variables
        self.assertTrue(np.allclose(self.beta_c, self.beta_python))

        # gammas / state probabilities
        self.assertTrue(np.allclose(self.gamma_c, self.gamma_python))

        # state counts
        self.assertTrue(np.allclose(self.statecount_c, self.statecount_python))

        # transition counts
        self.assertTrue(np.allclose(self.C_c, self.C_python))

        # viterbi
        #self.assertTrue(np.allclose(self.vpath_c, self.vpath_python))


if __name__=="__main__":
    unittest.main()
