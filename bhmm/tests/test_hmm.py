#!/usr/local/bin/env python

"""
Test HMM functions.

"""

import numpy as np
import scipy.sparse.linalg

from bhmm import testsystems

from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

def test_hmm():
    # Create a simple HMM model.
    model = testsystems.three_state_model()
    # Test model parameter access.
    assert_equal(model.Tij.shape, (3,3))
    assert_equal(model.Pi.shape, (3,))
    assert_equal(model.logPi.shape, (3,))

    return

def test_two_state_model():
    """Test the creation of a simple two-state HMM model with analytical parameters.
    """
    from bhmm import HMM
    # Create a simple two-state model.
    nstates = 2
    Tij = testsystems.generate_transition_matrix(reversible=True)
    states = [ {'mu' : -1, 'sigma' : 1}, {'mu' : +1, 'sigma' : 1} ]
    model = HMM(nstates, Tij, states)
    # Compute stationary probability using ARPACK.
    from scipy.sparse.linalg import eigs
    from numpy.linalg import norm
    [eigenvalues, eigenvectors] = eigs(Tij.T, k=1, which='LR')
    eigenvectors = np.real(eigenvectors)
    Pi = eigenvectors[:,0] / eigenvectors[:,0].sum()
    # Test model is correct.
    assert_array_almost_equal(model.Tij, Tij)
    assert_equal(model.states, states)
    assert_array_almost_equal(model.Pi, Pi)

