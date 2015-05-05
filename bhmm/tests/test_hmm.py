#!/usr/local/bin/env python

"""
Test HMM functions.

"""

import numpy as np
import unittest

import bhmm
from bhmm.util import testsystems

from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class TestHMM(unittest.TestCase):

    def test_hmm(self):
        # Create a simple HMM model.
        model = testsystems.dalton_model(nstates=3)
        # Test model parameter access.
        assert_equal(model.transition_matrix.shape, (3,3))
        assert_equal(model.stationary_distribution.shape, (3,))

        return

    def test_two_state_model(self):
        """Test the creation of a simple two-state HMM model with analytical parameters.
        """
        from bhmm import HMM
        # Create a simple two-state model.
        nstates = 2
        Tij = testsystems.generate_transition_matrix(reversible=True)
        from bhmm import GaussianOutputModel
        means=[-1,+1]
        sigmas=[1,1]
        output_model = GaussianOutputModel(nstates, means=means, sigmas=sigmas)
        model = bhmm.HMM(Tij, output_model)
        # Compute stationary probability using ARPACK.
        from scipy.sparse.linalg import eigs
        from numpy.linalg import norm
        [eigenvalues, eigenvectors] = eigs(Tij.T, k=1, which='LR')
        eigenvectors = np.real(eigenvectors)
        Pi = eigenvectors[:,0] / eigenvectors[:,0].sum()
        # Test model is correct.
        assert_array_almost_equal(model._Tij, Tij)
        assert_array_almost_equal(model._Pi, Pi)
        assert(np.allclose(model.output_model.means, np.array(means)))
        assert(np.allclose(model.output_model.sigmas, np.array(sigmas)))


if __name__=="__main__":
    unittest.main()
