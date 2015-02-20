__author__ = 'noe'

import numpy as np
from scipy import linalg

def eigenvalues(A, n):
    """
    Helper function for computing the eigenvalues of A in a sorted way. Should be replaced by EMMA function

    :param A:
    :param n:
    :return:
    """
    v=linalg.eigvals(A).real
    idx=(-v).argsort()[:n]
    return v[idx]

def stationary_distribution(P):
    """
    Computes the stationary distribution of P. Should be replaced by EMMA function

    :param P:
    :return:
    """
    (v,w) = linalg.eig(P.T)
    mu = w[:,np.argmax(v)]
    return mu / np.sum(mu)

    # @classmethod
    # def _compute_stationary_probabilities(cls, Tij, tol=1e-5, maxits=None, method='arpack'):
    #     """Compute the stationary probabilities for a given transition matrix.
    #
    #     Parameters
    #     ----------
    #     Tij : numpy.array with shape (nstates, nstates)
    #         The row-stochastic transition matrix for which the stationary probabilities are to be computed.
    #     tol : float, optional, default=1e-5
    #         The absolute tolerance in total variation distance between probability vector iterates at which iterations are terminated.
    #     maxits : int, optional, default=None
    #         If not None, the maximum number of iterations to perform.
    #     method : str, optional, default='arpack'
    #         Method of stationary eigenvector computation: ['arpack', 'inverse-iteration']
    #
    #     Returns
    #     -------
    #     Pi : numpy.array with shape (nstates, )
    #         The stationary probabilities corresponding to the row-stochastic transition matrix Tij.
    #
    #     Notes
    #     -----
    #     This function uses the inverse iteration: http://en.wikipedia.org/wiki/Inverse_iteration
    #
    #     Examples
    #     --------
    #
    #     Compute stationary probabilities for a given transition matrix.
    #
    #     >>> from bhmm import testsystems
    #     >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
    #     >>> Pi = HMM._compute_stationary_probabilities(Tij)
    #
    #     """
    #     nstates = Tij.shape[0]
    #
    #     if nstates == 2:
    #         # Use analytical method for 2x2 matrices.
    #         return np.array([Tij[1,0], Tij[0,1]], np.float64)
    #
    #     # For larger matrices, solve numerically.
    #     if method == 'arpack':
    #         # Compute stationary probability using ARPACK.
    #         # TODO: Pass 'maxits' and 'tol' to ARPACK?
    #         from scipy.sparse.linalg import eigs
    #         from numpy.linalg import norm
    #         [eigenvalues, eigenvectors] = eigs(Tij.T, k=1, which='LR')
    #         eigenvectors = np.real(eigenvectors)
    #         Pi = eigenvectors[:,0] / eigenvectors[:,0].sum()
    #         return Pi
    #     elif method == 'inverse-iteration':
    #         T = np.array(Tij, dtype=np.float64) # Promote matrix to float64
    #         mu = 1.0 # eigenvalue corresponding to eigenvector to extract
    #         I = np.eye(nstates, dtype=np.float64) # identity matrix
    #         b_old = np.ones([nstates], dtype=np.float64) / float(nstates) # initial eigenvector guess
    #
    #         # Perform inverse iteration.
    #         converged = False
    #         iteration = 1
    #         while not converged:
    #             # Update eigenvector guess
    #             b_new = np.dot(np.linalg.inv((T - mu*I).T), b_old)
    #
    #             # Normalize to be a probability.
    #             b_new /= b_new.sum()
    #
    #             # Compute total variation probability difference.
    #             delta = 0.5 * np.absolute(b_new - b_old).sum()
    #
    #             # Check convergence criterion.
    #             if maxits:
    #                 converged = (iteration >= maxits)
    #             if tol:
    #                 converged = (delta < tol)
    #
    #             # Proceed to next iteration.
    #             if not converged:
    #                 iteration += 1
    #                 b_old = b_new
    #
    #         # Normalize vector to sum to unity.
    #         Pi = b_new / b_new.sum()
    #         return Pi
    #     else:
    #         raise Exception("method %s unknown." % method)
    #
    #     return
