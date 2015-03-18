"""
Hidden Markov model

"""

import copy
import time
import numpy as np

import pyemma.msm.estimation as msmest
from numpy.linalg import norm

from bhmm import HMM
from ml.baum_welch import BaumWelchHMM

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

class MLHMM(object):
    """
    Maximum likelihood Hidden Markov model (HMM).

    This class is used to fit a maximum-likelihood HMM to data.

    Examples
    --------

    >>> from bhmm import testsystems
    >>> [model, O, S] = testsystems.generate_synthetic_observations()
    >>> mlhmm = MLHMM(O, model.nstates)
    >>> model = mlhmm.fit()

    """
    def __init__(self, observations, nstates, initial_model=None, reversible=True, verbose=False, output_model_type='gaussian',
                 kernel = 'c'):
        """Initialize a Bayesian hidden Markov model sampler.

        Parameters
        ----------
        observations : list of numpy arrays representing temporal data
            `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
        nstates : int
            The number of states in the model.
        initial_model : HMM, optional, default=None
            If specified, the given initial model will be used to initialize the BHMM.
            Otherwise, a heuristic scheme is used to generate an initial guess.
        reversible : bool, optional, default=True
            If True, a prior that enforces reversible transition matrices (detailed balance) is used;
            otherwise, a standard  non-reversible prior is used.
        verbose : bool, optional, default=False
            Verbosity flag.
        output_model_type : str, optional, default='gaussian'
            Output model type.  ['gaussian', 'discrete']
        kernel: str, optional, default='python'
            Implementation kernel

        """
        # Store options.
        self.verbose = verbose
        self.reversible = reversible

        # Store the number of states.
        self.nstates = nstates

        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        # Determine number of observation trajectories we have been given.
        self.ntrajectories = len(self.observations)

        if initial_model:
            # Use user-specified initial model, if provided.
            self.model = copy.deepcopy(initial_model)
        else:
            # Generate our own initial model.
            self.model = self._generateInitialModel(output_model_type)

        self.kernel = kernel

        return

    def fit(self):
        """Fit a maximum-likelihood HMM model.

        Returns
        -------
        model : HMM
            The maximum likelihood HMM model.


        Examples
        --------

        >>> from bhmm import testsystems
        >>> [model, O, S] = testsystems.generate_synthetic_observations()
        >>> mlhmm = MLHMM(O, model.nstates)
        >>> model = mlhmm.fit()

        """
        if self.verbose:
            print "================================================================================"
            print "Running Baum-Welch::"
            print "  input observations:"
            print self.observations
            print "  initial HMM guess:"
            print self.model

        initial_time = time.time()

        # Run Baum-Welch EM algorithm to fit the HMM.
        baumwelch = BaumWelchHMM(self.observations, self.model, kernel=self.kernel)
        self.model = baumwelch.fit()

        final_time = time.time()
        elapsed_time = final_time - initial_time

        if self.verbose:
            print "maximum likelihood HMM:"
            print str(self.model)
            print "Elapsed time for Baum-Welch solution: %.3f s" % elapsed_time
            print ""
            print "Computing Viterbi path:"

        initial_time = time.time()

        # Compute hidden state trajectories using the Viterbi algorithm.
        self.hidden_state_trajectories = baumwelch.viterbi_paths()

        final_time = time.time()
        elapsed_time = final_time - initial_time

        if self.verbose:
            print "Elapsed time for Viterbi path computation: %.3f s" % elapsed_time
            print "================================================================================"

        return self.model

    @classmethod
    def _transitionMatrixMLE(cls, Cij, reversible=True, maxits=1000, reltol=1e-5, epsilon=1e-3, verbose=False):
        """Compute maximum likelihood estimator of transition matrix from fractional transition counts.

        Parameters
        ----------
        Cij : np.array with size (nstates,nstates)
            The transition count matrix.
        reversible : bool, optional, default=True
            If True, the reversible transition matrix will be estimated.
        maxits : int, optional, default=1000
            The maximum number of iterations.
        reltol : float, optional, default=1e-5
            The relative tolerance.
        verbose : bool, optional, default=False
            If True, verbose output will be printed.

        References
        ----------
        [1] Prinz JH, Wu H, Sarich M, Keller B, Fischbach M, Held M, Chodera JD, Schuette C, and Noe F.
        Markov models of molecular kinetics: Generation and validation. JCP 134:174105, 2011.

        Examples
        --------

        >>> Cij = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        Non-reversible estimate.

        >>> Tij = MLHMM._transitionMatrixMLE(Cij, reversible=False)

        Reversible estimate.

        >>> Tij = MLHMM._transitionMatrixMLE(Cij, reversible=True)

        """
        # Determine size of count matrix.
        nstates = Cij.shape[0]

        # Ensure count matrix is double precision.
        Cij = np.array(Cij, dtype=np.float64)
        Tij = msmest.transition_matrix(Cij, reversible=reversible)

        return Tij

    def _generateInitialModel(self, output_model_type):
        """Use a heuristic scheme to generate an initial model.

        Parameters
        ----------
        output_model_type : str, optional, default='gaussian'
            Output model type.  ['gaussian', 'discrete']

        TODO
        ----
        * Replace this with EM or MLHMM procedure from Matlab code.

        """
        if output_model_type == 'discrete':
            import bhmm.init.discrete
            return bhmm.init.discrete.initial_model_discrete(self.observations, self.nstates, lag = 1, reversible = True)
        elif output_model_type == 'gaussian':
            import bhmm.init.gaussian
            return bhmm.init.gaussian.initial_model_gaussian1d(self.observations, self.nstates, reversible=True)
        else:
            raise NotImplementedError('output model type '+str(output_model_type)+' not yet implemented.')
