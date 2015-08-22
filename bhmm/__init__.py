#!/usr/local/bin/env python

"""
BHMM: A toolkit for Bayesian hidden Markov model analysis of single-molecule trajectories.

"""
from __future__ import absolute_import

# import API
from bhmm.api import *

# hmms
from bhmm.hmm.generic_hmm import HMM
from bhmm.hmm.gaussian_hmm import GaussianHMM
from bhmm.hmm.discrete_hmm import DiscreteHMM

from bhmm.hmm.generic_sampled_hmm import SampledHMM
from bhmm.hmm.gaussian_hmm import SampledGaussianHMM
from bhmm.hmm.discrete_hmm import SampledDiscreteHMM

# estimators
from bhmm.estimators.bayesian_sampling import BayesianHMMSampler as BHMM
from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as MLHMM

# output models
from bhmm.output_models import OutputModel, GaussianOutputModel, DiscreteOutputModel

# other stuff
from bhmm.util import config
from bhmm.util import testsystems

from .version import get_versions
__version__ = get_versions()['version']
del get_versions
