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
        # test
        A = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import pyemma.msm.analysis as msmana
        msmana.is_transition_matrix(A)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))
        # A should be close to P
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
        # Test if model fit is close to reference. Note that we do not have an exact reference, so we cannot set the
        # tolerance in a rigorous way to test statistical significance. These are just sanity checks.
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import pyemma.msm.analysis as msmana
        msmana.is_transition_matrix(Tij)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))
        #if (B[0,0]<B[1,0]):
        #    B = B[np.array([1,0]),:]
        Tij_ref = np.array([[0.99, 0.01],
                            [0.01, 0.99]])
        Bref = np.array([[0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 0.5, 0.5]])
        assert(np.max(Tij-Tij_ref) < 0.01)
        assert(np.max(B-Bref) < 0.05 or np.max(B[[1,0]]-Bref) < 0.05)

    def test_discrete_6_3(self):
        # 4x4 transition matrix
        nstates = 3
        P = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                      [0.20, 0.79, 0.01, 0.00, 0.00, 0.00],
                      [0.00, 0.01, 0.84, 0.15, 0.00, 0.00],
                      [0.00, 0.00, 0.05, 0.94, 0.01, 0.00],
                      [0.00, 0.00, 0.00, 0.02, 0.78, 0.20],
                      [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import pyemma.msm.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.initial_model_discrete(dtrajs, nstates)
        # Test stochasticity and reversibility
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        import pyemma.msm.analysis as msmana
        msmana.is_transition_matrix(Tij)
        msmana.is_reversible(Tij)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))


if __name__=="__main__":
    unittest.main()
