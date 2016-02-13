#!/usr/local/bin/env python

"""
Test BHMM using simple analytical models.

"""

import unittest
import numpy as np
import bhmm
from os.path import abspath, join
from os import pardir

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class TestBHMMPathological(unittest.TestCase):

    def test_2state_rev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        mle = bhmm.estimate_hmm([obs], nstates=2, lag=1)
        # this will generate disconnected count matrices and should fail:
        with self.assertRaises(NotImplementedError):
            bhmm.bayesian_hmm([obs], mle, reversible=True, p0_prior=None, transition_matrix_prior=None)

    def test_2state_nonrev_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        mle = bhmm.estimate_hmm([obs], nstates=2, lag=1)
        sampled = bhmm.bayesian_hmm([obs], mle, reversible=False, nsample=2000,
                                    p0_prior='mixed', transition_matrix_prior='mixed')
        for i, s in enumerate(sampled.sampled_hmms):
            print i
            print s.transition_matrix
            print s.output_model.output_probabilities
            print
        print 'std: \n', sampled.transition_matrix_std
        assert np.all(sampled.transition_matrix_std[0] > 0)
        assert np.allclose(sampled.transition_matrix_std[1], [0, 0])

    def test_2state_rev_2step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=int)
        mle = bhmm.estimate_hmm([obs], nstates=2, lag=1)
        sampled = bhmm.bayesian_hmm([obs], mle, reversible=False, nsample=100,
                                    p0_prior='mixed', transition_matrix_prior='mixed')
        assert np.all(sampled.transition_matrix_std>0)



if __name__=="__main__":
    unittest.main()
