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

import unittest
from bhmm.util import testsystems

class TestTestSystems(unittest.TestCase):

    def test_transition_matrix(self):
        """Test example transition matrices.
        """
        Tij = testsystems.generate_transition_matrix(nstates=3, reversible=False)
        Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
        # TODO: Check Tij is proper row-stochastic matrix?
        return

    def test_three_state_model(self):
        """Test three-state model.
        """
        model = testsystems.dalton_model()
        # TODO: Check stationary probiblities are correct?
        return


if __name__=="__main__":
    unittest.main()
