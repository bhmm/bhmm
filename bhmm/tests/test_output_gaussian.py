__author__ = 'noe'

import numpy as np
import unittest
import time
from bhmm.output_models.gaussian import GaussianOutputModel

print_speedup = False

class TestOutputGaussian(unittest.TestCase):

    def setUp(self):
        nstates = 3
        means    = np.array([-0.5, 0.0, 0.5])
        sigmas = np.array([0.2, 0.2, 0.2])
        self.G = GaussianOutputModel(nstates, means=means, sigmas=sigmas)

        # random Gaussian samples
        self.obs = np.random.randn((10000))


    def tearDown(self):
        pass


    def test_p_obs(self):
        # compare results
        self.G.set_implementation('c')
        time1 = time.time()
        for i in range(10):
            p_c = self.G.p_obs(self.obs)
        time2 = time.time()
        t_c = time2-time1

        self.G.set_implementation('python')
        time1 = time.time()
        for i in range(10):
            p_p = self.G.p_obs(self.obs)
        time2 = time.time()
        t_p = time2-time1

        assert(np.allclose(p_c, p_p))

        # speed report
        if print_speedup:
            print('p_obs speedup c/python = '+str(t_p/t_c))



if __name__=="__main__":
    unittest.main()
