__author__ = 'noe'

"""
Abstract base class for HMM output model.

TODO
----
* Allow new derived classes to be registered and retrieved.

"""

__author__ = "John D. Chodera, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["John D. Chodera", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "John D. Chodera"
__email__="jchodera AT gmail DOT com"


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

