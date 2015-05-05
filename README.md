[![Build Status](https://travis-ci.org/bhmm/bhmm.png)](https://travis-ci.org/bhmm/bhmm)

# Bayesian hidden Markov models for analysis of single-molecule trajectory data

This project provides tools for estimating the number of metastable states, rate constants between the states, equilibrium populations, distributions characterizing the states, and distributions of these quantities from single-molecule data. This data could be FRET data, single-molecule pulling data, or any data where one or more observables are recorded as a function of time. A Hidden Markov Model (HMM) is used to interpret the observed dynamics, and a distribution of models that fit the data is sampled using Bayesian inference techniques and Markov chain Monte Carlo (MCMC), allowing for both the characterization of uncertainties in the model and modeling of the expected information gain by new experiments.

## Manifest
* `LICENSE` - full text of LGPL v3 license
* `matlab/` - Matlab code for BHMM
* `docs/` - documentation
* `manscripts/` - manuscript sources
* `references/` - collected references from manuscript

## Authors
* John D. Chodera <john.chodera@choderalab.org>
* Bettina Keller <bettina.keller@fu-berlin.de>
* Phillip J. Elms <elms@berkeley.edu>
* Frank Noé <frank.noe@fu-berlin.de>
* Christian M. Kaiser <kaiser.jhu.bio@gmail.com>
* Aaron Ewall-Wice
* Susan Marqusee
* Carlos Bustamante
* Nina Singhal Hinrichs <nshinrichs@uchicago.edu>
