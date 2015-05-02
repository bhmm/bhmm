#!/usr/local/bin/env python

"""
Test MLHMM.

"""

import unittest
from functools import partial
from bhmm import testsystems
from bhmm import MaximumLikelihoodEstimator

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class TestMLHMM(unittest.TestCase):
    pass
    # TODO: these tests don't test anything, fill with something meaningful

    # def run_mlhmm(nstates):
    #     """
    #     Run the MLHMM on synthetic data with the given number of states.
    #
    #     Parameters
    #     ----------
    #     nstates : int
    #         The number of states to test the BHMM with.
    #
    #     """
    #     # Generate synthetic observations.
    #     [model, S, O] = model.generate_synthetic_observation_trajectories(nstates=nstates)
    #     # Fit an MLHMM.
    #     mlhmm = mlhmm.MLHMM(O, nstates)
    #     model = mlhmm.fit()
    #
    #     return
    #
    # def test_bhmm_synthetic(self):
    #     """
    #     Test the BHMM model on synthetic datasets.
    #
    #     """
    #
    #     nstates_min = 2 # minimum number of states to test
    #     nstates_max = 8 # maximum number of states to test
    #
    #     for nstates in range(nstates_min, nstates_max):
    #         f = partial(run_bhmm, nstates)
    #         f.description = "Testing BHMM on synthetic data for %d states"
    #         yield f
    #
    #     return


if __name__=="__main__":
    unittest.main()
