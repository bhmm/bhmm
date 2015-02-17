#!/usr/local/bin/env python

"""
Test provided HMM test system models.

"""

def test_transition_matrix():
    """Test example transition matrices.
    """
    from bhmm import testsystems
    Tij = testsystems.generate_transition_matrix(nstates=3, reversible=False)
    Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
    # TODO: Check Tij is proper row-stochastic matrix?
    return

def test_three_state_model():
    """Test three-state model.
    """
    from bhmm import testsystems
    model = testsystems.three_state_model()
    # TODO: Check stationary probiblities are correct?
    return
