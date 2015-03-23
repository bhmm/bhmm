__author__ = 'noe'

import numpy as np
import unittest
import bhmm.init.discrete as initdisc


class TestHMM(unittest.TestCase):

    def test_discrete_2_2(self):
        # 2x2 transition matrix
        P = np.array([[0.99,0.01],[0.01,0.99]])
        # generate realization
        import pyemma.msm.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.initial_model_discrete(dtrajs, 2)
        # A should be close to P
        A = hmm.Tij
        B = hmm.output_model.B
        if (B[0,0]<B[1,0]):
            B = B[np.array([1,0]),:]
        assert(np.max(A-P) < 0.01)
        assert(np.max(B-np.eye(2)) < 0.01)

    def test_discrete_4_2(self):
        # 4x4 transition matrix
        nstates = 2
        P = np.array([[0.90, 0.10, 0.00, 0.00],
                      [0.10, 0.89, 0.01, 0.00],
                      [0.00, 0.01, 0.89, 0.10],
                      [0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import pyemma.msm.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.initial_model_discrete(dtrajs, nstates)
        # Test if model fit is close to reference.
        # TODO: Set tolerance to ensure failures are statistically significant.
        Tij = hmm.Tij
        B = hmm.output_model.B
        #if (B[0,0]<B[1,0]):
        #    B = B[np.array([1,0]),:]
        Tij_ref = np.array([[0.99, 0.01],
                            [0.01, 0.99]])
        Bref = np.array([[0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 0.5, 0.5]])
        assert(np.max(Tij-Tij_ref) < 0.01)
        assert(np.max(B-Bref) < 0.05)

if __name__=="__main__":
    unittest.main()
