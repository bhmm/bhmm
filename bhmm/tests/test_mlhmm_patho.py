__author__ = 'noe'

import unittest
import numpy as np
import bhmm

class TestMLHMM_Pathologic(unittest.TestCase):

    def test_1state(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=1, lag=1, accuracy=1e-6)
        p0_ref = np.array([1.0])
        A_ref = np.array([[1.0]])
        B_ref = np.array([[1.0]])
        assert np.allclose(hmm.initial_distribution, p0_ref)
        assert np.allclose(hmm.transition_matrix, A_ref)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref)

    def test_1state_fail(self):
        obs = np.array([0, 0, 0, 0, 0], dtype=int)
        with self.assertRaises(NotImplementedError):
            bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)

    def test_2state_step(self):
        obs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.8, 0.2],
                          [0.0, 1.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_model.output_probabilities, B_ref[[perm]], atol=1e-5)

    def test_2state_2step(self):
        obs = np.array([0, 1, 0], dtype=int)
        hmm = bhmm.estimate_hmm([obs], nstates=2, lag=1, accuracy=1e-6)
        p0_ref = np.array([1, 0])
        A_ref = np.array([[0.0, 1.0],
                          [1.0, 0.0]])
        B_ref = np.array([[1, 0],
                          [0, 1]])
        perm = [1, 0]  # permutation
        assert np.allclose(hmm.initial_distribution, p0_ref, atol=1e-5) \
               or np.allclose(hmm.initial_distribution, p0_ref[perm], atol=1e-5)
        assert np.allclose(hmm.transition_matrix, A_ref, atol=1e-5) \
               or np.allclose(hmm.transition_matrix, A_ref[np.ix_(perm, perm)], atol=1e-5)
        assert np.allclose(hmm.output_model.output_probabilities, B_ref, atol=1e-5) \
               or np.allclose(hmm.output_model.output_probabilities, B_ref[[perm]], atol=1e-5)


if __name__=="__main__":
    unittest.main()
