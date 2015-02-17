"""
Bayesian hidden Markov models.

"""

import copy

class BHMM(object):
    """Bayesian hidden Markov model sampler.

    """
    def __init__(observations):
        """Initialize a Bayesian hidden Markov model sampler.

        Parameters
        ----------
        observations : list of 1d numpy arrays
            `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`

        """
        # Store a copy of the observations.
        self.observations = copy.deepcopy(observations)

        return

