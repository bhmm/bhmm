#!/usr/local/bin/env python

"""
Test MLHMM.

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

class TestMLHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        testfile = abspath(join(abspath(__file__), pardir))
        testfile = join(testfile, 'data')
        testfile = join(testfile, '2well_traj_100K.dat')
        obs = np.loadtxt(testfile, dtype=int)

        # don't print
        bhmm.config.verbose = False

        # hidden states
        nstates = 2

        # run with lag 1 and 10
        cls.hmm_lag1 = bhmm.estimate_hmm([obs], nstates, lag=1, type='discrete')
        cls.hmm_lag10 = bhmm.estimate_hmm([obs], nstates, lag=10, type='discrete')

    # =============================================================================
    # Test
    # =============================================================================

    def test_output_model(self):
        from bhmm import DiscreteOutputModel
        assert isinstance(self.hmm_lag1.output_model, DiscreteOutputModel)
        assert isinstance(self.hmm_lag10.output_model, DiscreteOutputModel)

    def test_reversible(self):
        assert self.hmm_lag1.is_reversible
        assert self.hmm_lag10.is_reversible

    def test_stationary(self):
        assert self.hmm_lag1.is_stationary
        assert self.hmm_lag10.is_stationary

    def test_lag(self):
        assert self.hmm_lag1.lag == 1
        assert self.hmm_lag10.lag == 10

    def test_nstates(self):
        assert self.hmm_lag1.nstates == 2
        assert self.hmm_lag10.nstates == 2

    def test_transition_matrix(self):
        import pyemma.msm.analysis as msmana
        for P in [self.hmm_lag1.transition_matrix, self.hmm_lag1.transition_matrix]:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_eigenvalues(self):
        for ev in [self.hmm_lag1.eigenvalues, self.hmm_lag10.eigenvalues]:
            assert len(ev) == 2
            assert np.isclose(ev[0], 1)
            assert ev[1] < 1.0

    def test_eigenvectors_left(self):
        for evec in [self.hmm_lag1.eigenvectors_left, self.hmm_lag10.eigenvectors_left]:
            assert np.array_equal(evec.shape, (2,2))
            assert np.sign(evec[0,0]) == np.sign(evec[0,1])
            assert np.sign(evec[1,0]) != np.sign(evec[1,1])

    def test_eigenvectors_right(self):
        for evec in [self.hmm_lag1.eigenvectors_right, self.hmm_lag10.eigenvectors_right]:
            assert np.array_equal(evec.shape, (2,2))
            assert np.isclose(evec[0,0], evec[1,0])
            assert np.sign(evec[0,1]) != np.sign(evec[1,1])

    def test_initial_distribution(self):
        for mu in [self.hmm_lag1.initial_distribution, self.hmm_lag10.initial_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # positivity
            assert np.all(mu > 0.0)
            # this data: approximately equal probability
            assert np.max(np.abs(mu[0]-mu[1])) < 0.05

    def test_stationary_distribution(self):
        for mu in [self.hmm_lag1.stationary_distribution, self.hmm_lag10.stationary_distribution]:
            # normalization
            assert np.isclose(mu.sum(), 1.0)
            # positivity
            assert np.all(mu > 0.0)
            # this data: approximately equal probability
            assert np.max(np.abs(mu[0]-mu[1])) < 0.05

    def test_lifetimes(self):
        for l in [self.hmm_lag1.lifetimes, self.hmm_lag10.lifetimes]:
            assert len(l) == 2
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        assert np.max(np.abs(self.hmm_lag10.lifetimes - 680)) < 20.0

    def test_timescales(self):
        for l in [self.hmm_lag1.timescales, self.hmm_lag10.timescales]:
            assert len(l) == 1
            assert np.all(l > 0.0)
        # this data: lifetimes about 680
        assert np.abs(self.hmm_lag10.timescales[0] - 340) < 20.0

if __name__=="__main__":
    unittest.main()
