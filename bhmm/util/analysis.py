"""
Analysis of BHMM data.

"""

import numpy as np

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"

def beta_confidence_intervals(ci_X, ntrials, ci=0.95):
    """
    Compute confidence intervals of beta distributions.

    Parameters
    ----------
    ci_X : numpy.array
        Computed confidence interval estimate from `ntrials` experiments
    ntrials : int
        The number of trials that were run.
    ci : float, optional, default=0.95
        Confidence interval to report (e.g. 0.95 for 95% confidence interval)

    Returns
    -------
    Plow : float
        The lower bound of the symmetric confidence interval.
    Phigh : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------

    >>> ci_X = np.random.rand(10,10)
    >>> ntrials = 100
    >>> [Plow, Phigh] = beta_confidence_intervals(ci_X, ntrials)

    """
    # Compute low and high confidence interval for symmetric CI about mean.
    ci_low = 0.5 - ci/2;
    ci_high = 0.5 + ci/2;

    # Compute for every element of ci_X.
    from scipy.stats import beta
    Plow = ci_X * 0.0;
    Phigh = ci_X * 0.0;
    for i in range(ci_X.shape[0]):
        for j in range(ci_X.shape[1]):
            Plow[i,j] = beta.ppf(ci_low, a = ci_X[i,j] * ntrials, b = (1-ci_X[i,j]) * ntrials);
            Phigh[i,j] = beta.ppf(ci_high, a = ci_X[i,j] * ntrials, b = (1-ci_X[i,j]) * ntrials);

    return [Plow, Phigh]

def empirical_confidence_interval(sample, interval=0.95):
    """
    Compute specified symmetric confidence interval for empirical sample.

    Parameters
    ----------
    sample : numpy.array
        The empirical samples.
    interval : float, optional, default=0.95
        Size of desired symmetric confidence interval (0 < interval < 1)
        e.g. 0.68 for 68% confidence interval, 0.95 for 95% confidence interval

    Returns
    -------
    low : float
        The lower bound of the symmetric confidence interval.
    high : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------

    >>> sample = np.random.randn(1000)
    >>> [low, high] = empirical_confidence_interval(sample)

    >>> [low, high] = empirical_confidence_interval(sample, interval=0.65)

    >>> [low, high] = empirical_confidence_interval(sample, interval=0.99)

    """

    # Sort sample in increasing order.
    sample = np.sort(sample)

    # Determine sample size.
    N = len(sample)

    # Compute low and high indices.
    low_index = int(np.round((N-1) * (0.5 - interval/2))) + 1
    high_index = int(np.round((N-1) * (0.5 + interval/2))) + 1

    # Compute low and high.
    low = sample[low_index]
    high = sample[high_index]

    return [low, high]

def generate_latex_table(sampled_hmm, sampling_time='1 ms', obs_name='force', obs_units='pN', conf=0.95):
    """
    Generate a LaTeX column-wide table showing various computed properties and uncertainties.

    """
    # check input
    from bhmm.hmm.gaussian_hmm import SampledGaussianHMM
    assert isinstance(sampled_hmm, SampledGaussianHMM), 'sampled_hmm ist not a SampledGaussianHMM'
    # confidence interval
    sampled_hmm.set_confidence(conf)
    # nstates
    nstates = sampled_hmm.nstates

    table = """
            \\begin{tabular*}{\columnwidth}{@{\extracolsep{\\fill}}lcc}
            \\hline
            \\multicolumn{2}{l}{\\bf Property} & \\bf Value\\\\ \\hline
            """
    # Stationary probability.
    p = sampled_hmm.stationary_distribution_mean
    p_lo, p_hi = sampled_hmm.stationary_distribution_conf
    for i in range(nstates):
        if (i == 0):
            table += 'Equilibrium probability '
        table += '& $\pi_{%d}$ & $%0.3f_{\:%0.3f}^{\:%0.3f}$ \\' % (i, p[i], p_lo[i], p_hi[i]) + '\n'
        table += '\\hline' + '\n'

    # Transition probabilities.
    P = sampled_hmm.transition_matrix_mean
    P_lo, P_hi = sampled_hmm.transition_matrix_conf
    for i in range(nstates):
        for j in range(nstates):
            if (i == 0) and (j==0):
                table += 'Transition probability ($\Delta t = $%s) ' % (sampling_time)
            table += '& $T_{%d%d}$ & $%f_{\:%f}^{\:%f}$ \\' % (i, j, P[i,j], P_lo[i,j], P_hi[i,j]) + '\n'
            table += '\\hline' + '\n'

    # State mean forces.
    m = sampled_hmm.means_mean
    m_lo, m_hi = sampled_hmm.transition_matrix_conf
    for i in range(nstates):
        if (i == 0):
            table += 'State %s mean (%s) ' % (obs_name, obs_units)
        table += '& $\mu_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\' % (i, m[i], m_lo[i], m_hi[i]) + '\n'
        table += '\\hline' + '\n'

    # State force standard deviations.
    s = sampled_hmm.means_mean
    s_lo, s_hi = sampled_hmm.transition_matrix_conf
    for i in range(nstates):
        if (i == 0):
            table += 'State %s std dev (%s) ' % (obs_name, obs_units)
        table += '& $\mu_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\' % (i, s[i], s_lo[i], s_hi[i]) + '\n'
        table += '\\hline' + '\n'

    # Transition rates via pseudogenerator.
    K = np.array(P)
    K_lo = np.array(P_lo)
    K_hi = np.array(P_hi)
    for i in range(sampled_hmm.nsamples):
        K[i] -= np.eye(sampled_hmm.nstates)
        K_lo[i] -= np.eye(sampled_hmm.nstates)
        K_hi[i] -= np.eye(sampled_hmm.nstates)
    for i in range(nstates):
        for j in range(nstates):
            if (i == 0) and (j==0):
                table += 'Transition rate (s$^{-1}$) '
            table += '& $k_{%d%d}$ & $%.1f_{\:%.1f}^{\:%.1f}$ \\' % (i, j, K[i,j], K_lo[i,j], K_hi[i,j]) + '\n'
            table += '\hline' + '\n'

    # State mean lifetimes.
    l = sampled_hmm.lifetimes_mean
    l_lo, l_hi = sampled_hmm.lifetimes_conf
    for i in range(nstates):
        if (i == 0):
            table += 'State mean lifetime (%s) ' % sampling_time
        table += '& $\mu_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\' % (i, l[i], l_lo[i], l_hi[i]) + '\n'
        table += '\\hline' + '\n'

    # State relaxation timescales.
    for i in range(nstates-1):
        if (i == 0):
            table += 'Relaxation time (%s) ' % sampling_time
        table += '& $\mu_{%d}$ & $%.3f_{\:%.3f}^{\:%.3f}$ \\' % (i, l[i], l_lo[i], l_hi[i]) + '\n'
        table += '\\hline' + '\n'

    table +="""
            \\hline
            \\end{tabular*}
            \\end{table}
            """
    return table

