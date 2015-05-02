__author__ = 'noe'

from generic_hmm import HMM
from bhmm.output_models.discrete import DiscreteOutputModel

class DiscreteHMM(HMM, DiscreteOutputModel):
    r""" Convenience access to an HMM with a Gaussian output model.

    """

    def __init__(self, hmm):
        # superclass constructors
        if not isinstance(hmm.output_model, DiscreteOutputModel):
            raise TypeError('Given hmm is not a discrete HMM, but has an output model of type: '+
                            str(type(hmm.output_model)))
        DiscreteOutputModel.__init__(self, hmm.output_model.output_probabilities)
        HMM.__init__(self, hmm.transition_matrix, self, lag=hmm.lag, Pi=hmm.initial_distribution,
                     stationary=hmm.is_stationary, reversible=hmm.is_reversible)
