"""
Hidden Markov model

"""

import numpy as np

class HMM(object):
    """
    Hidden Markov model (HMM).

    This class is used to represent an HMM. This could be a maximum-likelihood HMM or a sampled HMM from a Bayesian posterior.

    Examples
    --------

    >>> nstates = 2
    >>> Tij = np.array([[0.8, 0.2], [0.5, 0.5]])
    >>> states = [ {'mu' : -1, 'sigma' : 1}, {'mu' : +1, 'sigma' : 1} ]
    >>> model = HMM(nstates, Tij, states)

    """
    def __init__(self, nstates, Tij, states, dtype=np.float64):
        """
        Parameters
        ----------
        nstates : int
            The number of discrete output states.
        Tij : np.array with shape (nstates, nstates), optional, default=None
            Row-stochastic transition matrix among states.
            If `None`, the identity matrix will be used.
        states : list of dict
            `states[i]` is a dict of parameters for state `i`, with Gaussian output parameters `mu` (mean) and `sigma` (standard deviation).

        """
        # TODO: Perform sanity checks on data consistency.

        self.Tij = Tij
        self.Pi = self._compute_stationary_probabilities(self.Tij)
        self.states = states

        return

    @property
    def logPi(self):
        return np.log(self.Pi)

    @classmethod
    def _compute_stationary_probabilities(cls, Tij, tol=1e-5, maxits=None, method='arpack'):
        """Compute the stationary probabilities for a given transition matrix.

        Parameters
        ----------
        Tij : numpy.array with shape (nstates, nstates)
            The row-stochastic transition matrix for which the stationary probabilities are to be computed.
        tol : float, optional, default=1e-5
            The absolute tolerance in total variation distance between probability vector iterates at which iterations are terminated.
        maxits : int, optional, default=None
            If not None, the maximum number of iterations to perform.
        method : str, optional, default='arpack'
            Method of stationary eigenvector computation: ['arpack', 'inverse-iteration']

        Returns
        -------
        Pi : numpy.array with shape (nstates, )
            The stationary probabilities corresponding to the row-stochastic transition matrix Tij.

        Notes
        -----
        This function uses the inverse iteration: http://en.wikipedia.org/wiki/Inverse_iteration

        Examples
        --------
        >>> import testsystems
        >>> Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
        >>> Pi = HMM._compute_stationary_probabilities(Tij)

        """
        nstates = Tij.shape[0]

        if method == 'arpack':
            # Compute stationary probability using ARPACK.
            # TODO: Pass 'maxits' and 'tol' to ARPACK?
            from scipy.sparse.linalg import eigs
            from numpy.linalg import norm
            [eigenvalues, eigenvectors] = eigs(Tij.T, k=1, which='LR')
            eigenvectors = np.real(eigenvectors)
            Pi = eigenvectors[:,0] / eigenvectors[:,0].sum()
            return Pi
        elif method == 'inverse-iteration':
            T = np.array(Tij, dtype=np.float64) # Promote matrix to float64
            mu = 1.0 # eigenvalue corresponding to eigenvector to extract
            I = np.eye(nstates, dtype=np.float64) # identity matrix
            b_old = np.ones([nstates], dtype=np.float64) / float(nstates) # initial eigenvector guess

            # Perform inverse iteration.
            converged = False
            iteration = 1
            while not converged:
                # Update eigenvector guess
                b_new = np.dot(np.linalg.inv((T - mu*I).T), b_old)

                # Normalize to be a probability.
                b_new /= b_new.sum()

                # Compute total variation probability difference.
                delta = 0.5 * np.absolute(b_new - b_old).sum()

                # Check convergence criterion.
                if maxits:
                    converged = (iteration >= maxits)
                if tol:
                    converged = (delta < tol)

                # Proceed to next iteration.
                if not converged:
                    iteration += 1
                    b_old = b_new

            # Normalize vector to sum to unity.
            Pi = b_new / b_new.sum()
            return Pi
        else:
            raise Exception("method %s unknown." % method)

        return
