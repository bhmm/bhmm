__author__ = 'noe'


def generate_initial_model(observations, nstates, output_model_type):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    output_model_type : str, optional, default='gaussian'
        Output model type.  ['gaussian', 'discrete']

    TODO
    ----
    * Replace this with EM or MLHMM procedure from Matlab code.

    """
    if output_model_type == 'discrete':
        import bhmm.init.discrete
        return bhmm.init.discrete.initial_model_discrete(observations, nstates, lag = 1, reversible = True)
    elif output_model_type == 'gaussian':
        import bhmm.init.gaussian
        return bhmm.init.gaussian.initial_model_gaussian1d(observations, nstates, reversible=True)
    else:
        raise NotImplementedError('output model type '+str(output_model_type)+' not yet implemented.')

