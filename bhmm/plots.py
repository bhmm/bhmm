"""
Plotting utilities for Bayesian hidden Markov models.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_state_assignments(model, s_t, o_t, tau=1.0, time_units=None, obs_label=None, title=None, figsize=(7.5,1.5), markersize=3, pdf_filename=None):
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
    tau : float, optional, default=1.0
        Time interval between samples.
    time_units : str, optional, default=None
        Time unit label to associate with temporal axis.
    obs_label : str, optional, default=None
        Observation axis label.
    title : str, optional, default=None
        Title for the plot, if desired.
    pdf_filename : str, optional, default=None
        If specified, the plot will be written to a PDF file.

    Example
    -------

    >>> import tempfile
    >>> filename = tempfile.NamedTemporaryFile().name
    >>> from bhmm import testsystems
    >>> model = testsystems.dalton_model(nstates=3)
    >>> [model, O, S, bhmm] = testsystems.generate_random_bhmm(nstates=3, ntrajectories=1, length=10000)
    >>> models = bhmm.sample(nsamples=1, save_hidden_state_trajectory=True)
    >>> plot_state_assignments(model, models.hidden_state_trajectories[0], O[0], pdf_filename=filename)

    Label the axes.

    >>> plot_state_assignments(model, models.hidden_state_trajectories[0], O[0], tau=0.001, time_units='ms', obs_label='force / pN', pdf_filename=filename)

    """
    if pdf_filename:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(pdf_filename)

    # Set plotting style.
    np.random.seed(sum(map(ord, "aesthetics")))
    palette = sns.color_palette('muted', model.nstates)
    sns.set(style='white', palette=palette, font_scale=0.75)

    # Create subplots.
    gridspec = { 'width_ratios' : [0.85, 0.15] }
    f, axes = plt.subplots(1,2, figsize=figsize, sharey=True, gridspec_kw=gridspec)
    ax1 = axes[0]; ax2 = axes[1]
    #f.tight_layout()
    f.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, wspace=0.0)
    #f.subplots_adjust(left=0.2, wspace=0.6)
    #ax1 = plt.subplot2grid((1,10), (0, 0), colspan=9, figsize=figsize)
    #ax2 = plt.subplot2grid((1,10), (0, 9), colspan=1, figsize=figsize)

    ax1.hold(True)
    ax2.hold(True)

    # Determine min and max of observable range.
    omin = o_t.min(); omax = o_t.max()
    tmin = 0; tmax = o_t.size * tau

    # Plot.
    colors = list()
    npoints=100 # number of points per emission plot
    nbins = 40 # number of bins for histograms
    for state_index in range(model.nstates):
        # Find and plot samples in state.
        indices = np.where(s_t == state_index)
        tvec = tau * np.array(np.squeeze(indices), dtype=np.float32)
        line, = ax1.plot(tvec, o_t[indices], '.', markersize=markersize)
        color = line.get_color() # extract line color for additional plots
        colors.append(color)
        # Plot shading at one standard deviation width.
        state = model.states[state_index]
        if state['model'] == 'gaussian':
            mu = model.states[state_index]['mu']
            sigma = model.states[state_index]['sigma']
        else:
            # TODO: Generalize this to other kinds of output models.
            raise Exception('Not supported for non-gaussian output models.')
        ax1.plot(np.array([tmin, tmax]), mu*np.ones([2]), color=color, linewidth=1, alpha=0.7)
        ax1.fill_between(np.array([tmin, tmax]), (mu-sigma)*np.ones([2]), (mu+sigma)*np.ones([2]), facecolor=color, alpha=0.3, linewidth=0)
        # Plot histogram.
        ax2.hist(o_t[indices], nbins, align='mid', orientation='horizontal', color=color, stacked=True, edgecolor=None, alpha=0.5, linewidth=0, normed=True)
        #ovec = np.linspace(omin, omax, nbins)
        #N, b, p = plt.hist(o_t[indices], ovec)
        #dx = (ovec[-1]-ovec[0])/(nbins-1)
        #ovec = ovec[0:nbins]
        #pvec = N / N.sum() / dx
        #print ovec.shape
        #print pvec.shape
        #ax2.barh(pvec, ovec, color=color, align='center')
        # Plot state observable distribtuion.
        ovec = np.linspace(omin, omax, npoints)
        pvec = model.emission_probability(state_index, ovec)
        ax2.plot(pvec, ovec, color=color, linewidth=1)

    if title:
        ax1.set_title(title)

    # Label axes.
    xlabel = 'time'
    if time_units:
        xlabel += ' / %s' % time_units
    ylabel = 'observable'
    if obs_label:
        ylabel = obs_label
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    # Clear ticks for probability histogram plot.
    ax2.set_xticks([])

    # Despine
    sns.despine()

    if pdf_filename:
        # Write figure.
        pp.savefig()
        pp.close()

    return

def total_state_visits(nstates, S):
    """
    Return summary statistics for state trajectories.

    Parameters
    ----------
    nstates : int
        The number of states.
    S : list of numpy.array
        S[i] is the hidden state trajectory from state i

    """

    N_i = np.zeros([nstates], np.int32)
    min_state = nstates
    max_state = 0
    for s_t in S:
        for state_index in range(nstates):
            N_i[state_index] += (s_t == state_index).sum()
        min_state = min(min_state, s_t.min())
        max_state = max(max_state, s_t.max())
    return [N_i, min_state, max_state]

if __name__ == '__main__':
    # Create plots.
    from bhmm import testsystems
    [model, O, S, bhmm] = testsystems.generate_random_bhmm(nstates=3, ntrajectories=1, length=10000)
    models = bhmm.sample(nsamples=1, save_hidden_state_trajectory=True)
    # Extract hidden state trajectories and observations.
    s_t = S[0]
    o_t = O[0]

    # Plot.
    plot_state_assignments(model, s_t, o_t, pdf_filename='plot.pdf')

