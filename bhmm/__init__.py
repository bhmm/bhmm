#!/usr/local/bin/env python

"""
BHMM: A toolkit for Bayesian hidden Markov model analysis of single-molecule trajectories.

"""

# Define global version.
import version
__version__ = version.version

from bhmm.hmm.generic_hmm import HMM
from bhmm.estimators.bayesian_sampling import BHMM
from bhmm.estimators.maximum_likelihood import MLHMM

from output_models import OutputModel, GaussianOutputModel, DiscreteOutputModel

