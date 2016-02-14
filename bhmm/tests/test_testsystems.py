
# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
from bhmm.util import testsystems


class TestTestSystems(unittest.TestCase):

    def test_transition_matrix(self):
        """Test example transition matrices.
        """
        Tij = testsystems.generate_transition_matrix(nstates=3, reversible=False)
        Tij = testsystems.generate_transition_matrix(nstates=3, reversible=True)
        # TODO: Check Tij is proper row-stochastic matrix?
        return

    def test_three_state_model(self):
        """Test three-state model.
        """
        model = testsystems.dalton_model()
        # TODO: Check stationary probiblities are correct?
        return


if __name__ == "__main__":
    unittest.main()
