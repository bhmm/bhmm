__author__ = 'noe'

import numpy as np

from bhmm.hmm.generic_hmm import HMM
from bhmm.hmm.generic_sampled_hmm import SampledHMM
from bhmm.output_models.discrete import DiscreteOutputModel
from bhmm.util import config
from bhmm.util.statistics import confidence_interval_arr

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


class SampledDiscreteHMM(DiscreteHMM, SampledHMM):
    """ Sampled Discrete HMM with a representative single point estimate and error estimates

    Parameters
    ----------
    estimated_hmm : :class:`HMM <generic_hmm.HMM>`
        Representative HMM estimate, e.g. a maximum likelihood estimate or mean HMM.
    sampled_hmms : list of :class:`HMM <generic_hmm.HMM>`
        Sampled HMMs
    conf : float, optional, default = 0.95
        confidence interval, e.g. 0.68 for 1 sigma or 0.95 for 2 sigma.

    """
    def __init__(self, estimated_hmm, sampled_hmms, conf=0.95):
        # call GaussianHMM superclass constructer with estimated_hmm
        DiscreteHMM.__init__(self, estimated_hmm)
        # call SampledHMM superclass constructor
        SampledHMM.__init__(self, estimated_hmm, sampled_hmms, conf=conf)

    @property
    def output_probabilities_samples(self):
        r""" Samples of the output probability matrix """
        res = np.empty((self.nsamples, self.nstates, self.dimension), dtype=config.dtype)
        for i in range(self.nsamples):
            res[i,:,:] = self._sampled_hmms[i].means
        return res

    @property
    def output_probabilities_mean(self):
        r""" The mean of the output probability matrix """
        return np.mean(self.output_probabilities_samples, axis=0)

    @property
    def output_probabilities_std(self):
        r""" The standard deviation of the output probability matrix """
        return np.std(self.output_probabilities_samples, axis=0)

    @property
    def output_probabilities_conf(self):
        r""" The standard deviation of the output probability matrix """
        return confidence_interval_arr(self.output_probabilities_samples, conf=self._conf)
