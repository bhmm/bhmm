"""
Plotting utilities for Bayesian hidden Markov models.

"""

import numpy as np
import matplotlib as plt
import seaborn as sns

def plot_state_assignments(model, s_t, o_t, figsize=(8,3)):
    """
    Plot hidden state assignments and emission probabilities.

    Parameters
    ----------
    model : bhmm.HMM
        An HMM model with hidden state assignments to be plotted.
    s_t : numpy.array of shape (T) of int type
        The state trajectory to be plotted.
    o_t : numpy.array of shape (T) of float type
        The observation trajectory to be plotted.

    Example
    -------

    >>> from bhmm import testsystems
    >>> model = testsystems.dalton_model(nstates=3)
    >>> plot_state_assignments(model)

    """
    # Create plot.
    palette = sns.color_palette('muted', model.nstates)
    sns.set(style='darkgrid', palette=palette)
    f, axes = plt.subplots(1,2, figsize=figsize, sharey=True)

    # Plot.
    for state_index in range(model.nstates):
        indices = np.which(s_t == state_index)
        tvec = model.tau * indices
        axes.plot(tvec, o_t[indices], '.')

    return

# DEBUG
def total_state_visits(nstates, S):
    N_i = np.zeros([nstates], np.int32)
    min_state = nstates
    max_state = 0
    for s_t in S:
        for state_index in range(nstates):
            N_i[state_index] += (s_t == state_index).sum()
        min_state = min(min_state, s_t.min())
        max_state = max(max_state, s_t.max())
    print "min_state = %d, max_state = %d" % (min_state, max_state)
    return N_i

if __name__ == '__main__':
    # Test plotting to PDF.
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('plot.pdf')

    # Create plots.
    from bhmm import testsystems
    [model, O, S, bhmm] = testsystems.generate_random_bhmm(nstates=3, ntrajectories=1, length=10000)
    print model.Tij
    print S
    print total_state_visits(model.nstates, S)
    print O
    models = bhmm.sample(nsamples=1, save_hidden_state_trajectory=True)
    # Extract hidden state trajectories and observations.
    s_t = S[0]
    o_t = O[0]
    # Plot.
    plot_state_assignments(model, s_t, o_t)

    # Write figure.
    pp.savefig()
    pp.close()

