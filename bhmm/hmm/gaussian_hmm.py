__author__ = 'noe'

from generic_hmm import HMM
from bhmm.output_models.gaussian import GaussianOutputModel

class GaussianHMM(HMM, GaussianOutputModel):
    r""" Convenience access to an HMM with a Gaussian output model.

    """

    def __init__(self, hmm):
        # superclass constructors
        if not isinstance(hmm.output_model, GaussianOutputModel):
            raise TypeError('Given hmm is not a Gaussian HMM, but has an output model of type: '+
                            str(type(hmm.output_model)))
        GaussianOutputModel.__init__(self, hmm.nstates, means=hmm.output_model.means, sigmas=hmm.output_model.sigmas)
        HMM.__init__(self, hmm.transition_matrix, self, lag=hmm.lag, Pi=hmm.initial_distribution,
                     stationary=hmm.is_stationary, reversible=hmm.is_reversible)
