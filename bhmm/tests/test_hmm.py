#!/usr/local/bin/env python

"""
Test HMM functions.

"""

import numpy as np
import numpy.linalg as linalg

from nose.tools import ok_, eq_

def test_hmm():
    # Create a simple HMM model.
    from bhmm import testsystems
    model = testsystems.three_state_model()
    # Test model parameter access.
    eq_(model.Tij.shape, (3,3))
    eq_(model.Pi.shape, (3,))
    eq_(model.logPi.shape, (3,))

    return

def test_two_state_model():
    """Test the creation of a simple two-state HMM model with analytical parameters.
    """
    from bhmm import hmm
    # Create a simple two-state model.
    nstates = 2
    Tij = np.array([[0.8, 0.2], [0.5, 0.5]], np.float64)
    Pi = np.array([0.5, 0.5], np.float64)
    states = [ {'mu' : -1, 'sigma' : 1}, {'mu' : +1, 'sigma' : 1} ]
    model = hmm.HMM(nstates, Tij, states)
    # Test model is correct.
    eq_(linalg.norm(model.Tij - Tij), 0.0)
    eq_(model.states, states)
    eq_(linalg.norm(model.Pi - Pi), 0.0)

