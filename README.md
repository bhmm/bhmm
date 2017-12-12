[![Build Status](https://travis-ci.org/bhmm/bhmm.png?branch=master)](https://travis-ci.org/bhmm/bhmm)

# Bayesian hidden Markov model toolkit

This toolkit provides machinery for sampling from the Bayesian posterior of hidden Markov models with various choices of prior and output models.

## Installation

### Installation from conda

The easiest way to install `bhmm` is via the [`conda` package manager](http://conda.pydata.org/):
```
conda config --add channels conda-forge
conda install bhmm
```

### Installation from source

```
python setup.py install
```

## References

See [here](http://arxiv.org/abs/1108.1430) for a manuscript describing the theory behind using Gibbs sampling to sample from Bayesian hidden Markov model posteriors.

> Bayesian hidden Markov model analysis of single-molecule force spectroscopy: Characterizing kinetics under measurement uncertainty.
> John D. Chodera, Phillip Elms, Frank Noé, Bettina Keller, Christian M. Kaiser, Aaron Ewall-Wice, Susan Marqusee, Carlos Bustamante, Nina Singhal Hinrichs
> http://arxiv.org/abs/1108.1430

## Package maintainers
* Frank Noé <frank.noe@fu-berlin.de>, Freie Universität Berlin
* Martin K. Scherer <m.scherer@fu-berlin.de>, Freie Universität Berlin
* John D. Chodera <john.chodera@choderalab.org>, Sloan Kettering Institute
