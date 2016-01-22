__author__ = 'noe'

import unittest
import numpy as np
import bhmm

class TestMLHMM_Pathologic(unittest.TestCase):

    def test_2state_nonrev(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=2, lag=1, type='discrete')
        print 'HMM', hmm.transition_matrix

if __name__=="__main__":
    unittest.main()
