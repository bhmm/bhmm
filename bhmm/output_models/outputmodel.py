__author__ = 'noe'

"""
Abstract base class for HMM output model.

TODO
----
* Allow new derived classes to be registered and retrieved.

"""

import numpy as np

class OutputModel(object):
    """
    HMM output probability model abstract base class.

    """

    def __init__(self, nstates):
        """
        Create a general output model.

        Parameters
        ----------
        nstates : int
            The number of output states.

        """
        self.nstates = nstates

        return

