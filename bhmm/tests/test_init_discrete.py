__author__ = 'noe'

import numpy as np
import unittest
import bhmm.init.discrete as initdisc


class TestHMM(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------
    # Test correct initialization of metastable trajectories
    # ------------------------------------------------------------------------------------------------------

    def _test_discrete_2_2(self):
        # 2x2 transition matrix
        P = np.array([[0.99,0.01],[0.01,0.99]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.estimate_initial_model(dtrajs, 2)
        # test
        A = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import msmtools.analysis as msmana
        msmana.is_transition_matrix(A)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))
        # A should be close to P
        if (B[0,0]<B[1,0]):
            B = B[np.array([1,0]),:]
        assert(np.max(A-P) < 0.01)
        assert(np.max(B-np.eye(2)) < 0.01)

    def _test_discrete_4_2(self):
        # 4x4 transition matrix
        nstates = 2
        P = np.array([[0.90, 0.10, 0.00, 0.00],
                      [0.10, 0.89, 0.01, 0.00],
                      [0.00, 0.01, 0.89, 0.10],
                      [0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.estimate_initial_model(dtrajs, nstates)
        # Test if model fit is close to reference. Note that we do not have an exact reference, so we cannot set the
        # tolerance in a rigorous way to test statistical significance. These are just sanity checks.
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        # Test stochasticity
        import msmtools.analysis as msmana
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

    def _test_discrete_6_3(self):
        # 4x4 transition matrix
        nstates = 3
        P = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                      [0.20, 0.79, 0.01, 0.00, 0.00, 0.00],
                      [0.00, 0.01, 0.84, 0.15, 0.00, 0.00],
                      [0.00, 0.00, 0.05, 0.94, 0.01, 0.00],
                      [0.00, 0.00, 0.00, 0.02, 0.78, 0.20],
                      [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]])
        # generate realization
        import msmtools.generation as msmgen
        T = 10000
        dtrajs = [msmgen.generate_traj(P, T)]
        # estimate initial HMM with 2 states - should be identical to P
        hmm = initdisc.estimate_initial_model(dtrajs, nstates)
        # Test stochasticity and reversibility
        Tij = hmm.transition_matrix
        B = hmm.output_model.output_probabilities
        import msmtools.analysis as msmana
        msmana.is_transition_matrix(Tij)
        msmana.is_reversible(Tij)
        np.allclose(B.sum(axis=1), np.ones(B.shape[0]))

    # ------------------------------------------------------------------------------------------------------
    # Test correct initialization of pathological cases - single states, partial connectivity, etc.
    # ------------------------------------------------------------------------------------------------------

    def _test_1state_1obs(self):
        obs = np.array([0, 0, 0, 0, 0])
        Aref = np.array([[1.0]])
        Bref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = initdisc.estimate_initial_model([obs], 1, reversible=rev)
            assert(np.allclose(hmm.transition_matrix, Aref))
            assert(np.allclose(hmm.output_model.output_probabilities, Bref))

    # TODO: should also raise because there is only one metastable state.
    def _test_1state_2obs(self):
        obs = np.array([0, 0, 0, 0, 1])
        Aref = np.array([[1.0]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = initdisc.estimate_initial_model([obs], 1, reversible=rev)
            assert(np.allclose(hmm.transition_matrix, Aref))
            # output must be 1 x 2, and no zeros
            B = hmm.output_model.output_probabilities
            assert(np.array_equal(B.shape, np.array([1, 2])))
            assert(np.all(B > 0.0))

    def test_2state_2obs_unidirectional(self):
        obs = np.array([0, 0, 0, 0, 1])
        Aref = np.array([[ 0.75,  0.25],
                         [ 0.  ,  1.  ]])
        Bref = np.array([[ 1.,  0.],
                         [ 0.,  1.]])
        for rev in [True, False]:  # reversibiliy doesn't matter in this example
            hmm = initdisc.estimate_initial_model([obs], 2, reversible=rev)
            assert np.allclose(hmm.transition_matrix, Aref)
            assert np.allclose(hmm.output_model.output_probabilities, Bref)

    def _test_3state_fail(self):
        obs = np.array([0, 1, 2, 0, 3, 4])
        # this example doesn't admit more than 1 metastable state. Raise.
        with self.assertRaises(AssertionError):
            initdisc.estimate_initial_model([obs], 2, reversible=False)

    def _test_3state_prev(self):
        import msmtools.analysis as msmana
        obs = np.array([0, 1, 2, 0, 3, 4])
        # this example doesn't admit more than 1 metastable state. Raise.
        for rev in [True, False]:
            hmm = initdisc.estimate_initial_model([obs], 3, reversible=rev, prior_neighbor=0.01, prior_diag=0.01)
            assert msmana.is_transition_matrix(hmm.transition_matrix)
            if rev:
                assert msmana.is_reversible(hmm.transition_matrix)
            assert np.allclose(hmm.output_model.output_probabilities.sum(axis=1), 1)


if __name__=="__main__":
    unittest.main()
