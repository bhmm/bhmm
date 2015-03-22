__author__ = 'noe'


def generate_initial_model(observations, nstates, output_model_type, verbose=False):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    output_model_type : str, optional, default='gaussian'
        Output model type.  ['gaussian', 'discrete']
    verbose : bool, optional, default=False
        If True, will be verbose in output.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> [model, observations, states] = generate_synthetic_observations(output_model_type='gaussian')
    >>> initial_model = generate_initial_model(observations, model.nstates, 'gaussian')

    Generate initial model for a discrete output model.

    >>> [model, observations, states] = generate_synthetic_observations(output_model_type='discreten')
    >>> initial_model = generate_initial_model(observations, model.nstates, 'discrete')

    """
    if output_model_type == 'discrete':
        import bhmm.init.discrete
        return bhmm.init.discrete.initial_model_discrete(observations, nstates, lag=1, reversible=True, verbose=verbose)
    elif output_model_type == 'gaussian':
        import bhmm.init.gaussian
        return bhmm.init.gaussian.initial_model_gaussian1d(observations, nstates, reversible=True, verbose=verbose)
    else:
        raise NotImplementedError('output model type '+str(output_model_type)+' not yet implemented.')

