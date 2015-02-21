#!/usr/local/bin/env python

"""
Test provided HMM test system models.

"""

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"


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
    model = testsystems.dalton_model()
    # TODO: Check stationary probiblities are correct?
    return
