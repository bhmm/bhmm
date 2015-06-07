__author__ = 'noe'

import numpy as np
import math

from bhmm.msm import linalg

__author__ = "Hao Wu, Frank Noe"
__copyright__ = "Copyright 2015, John D. Chodera and Frank Noe"
__credits__ = ["Hao Wu", "Frank Noe"]
__license__ = "FreeBSD"
__maintainer__ = "Frank Noe"
__email__="frank.noe AT fu-berlin DOT de"


# some shortcuts
eps=np.spacing(0)
log=math.log
exp=math.exp

class TransitionMatrixSamplerRev:
    """
    Reversible transition matrix sampling using Hao Wu's new reversible sampling method.
    Automatically uses a -1 prior that ensures that maximum likelihood and mean are identical, i.e. you
    get error bars that are nicely envelopping the MLE.

    """

    def __init__(self, _C):
        """
        Initializes the transition matrix sampler with the observed count matrix

        Parameters:
        -----------
        C : ndarray(n,n)
            count matrix containing observed counts. Do not add a prior, because this sampler intrinsically
            assumes a -1 prior!

        """
        self.C = np.array(_C, dtype=np.float64)
        self.n = self.C.shape[0]
        self.sumC = self.C.sum(1)+0.0
        self.X = None
        # check input
        if np.min(self.sumC <= 0):
            raise ValueError('Count matrix has row sums of zero or less. Make sure that every state is visited!')


    def _is_positive(self, x):
        """
        Helper function, tests if x is numerically positive

        :param x:
        :return:
        """
        return x>=eps and (not math.isinf(x)) and (not math.isnan(x))


    def _update_step(self, v0, v1, v2, c0, c1, c2, random_walk_stepsize=1):
        """
        update the sample v0 according to
        the distribution v0^(c0-1)*(v0+v1)^(-c1)*(v0+v2)^(-c2)

        :param v0:
        :param v1:
        :param v2:
        :param c0:
        :param c1:
        :param c2:
        :param random_walk_stepsize:
        :return:
        """
        a = c1+c2-c0
        b = (c1-c0)*v2+(c2-c0)*v1
        c = -c0*v1*v2
        v_bar = 0.5*(-b+(b*b-4*a*c)**.5)/a
        h = c1/(v_bar + v1)**2 + c2/(v_bar + v2)**2 - c0/v_bar**2
        k = -h*v_bar*v_bar
        theta=-1/(h*v_bar)
        if self._is_positive(k) and self._is_positive(theta):
            v0_new = np.random.gamma(k,theta)
            if self._is_positive(v0_new):
                if v0 == 0:
                    v0 = v0_new
                else:
                    log_prob_new = (c0-1)*log(v0_new)-c1*log(v0_new+v1)-c2*log(v0_new+v2)
                    log_prob_new -= (k-1)*log(v0_new)-v0_new/theta
                    log_prob_old = (c0-1)*log(v0)-c1*log(v0+v1)-c2*log(v0+v2)
                    log_prob_old -= (k-1)*log(v0)-v0/theta
                    if np.random.rand()<exp(min(log_prob_new-log_prob_old,0)):
                        v0=v0_new
        v0_new = v0*exp(random_walk_stepsize*np.random.randn())
        if self._is_positive(v0_new):
            if v0 == 0:
                v0 = v0_new
            else:
                log_prob_new = c0*log(v0_new)-c1*log(v0_new+v1)-c2*log(v0_new+v2)
                log_prob_old = c0*log(v0)-c1*log(v0+v1)-c2*log(v0+v2)
                if np.random.rand() < exp(min(log_prob_new-log_prob_old,0)):
                    v0 = v0_new

        return v0


    def _update(self, n_step):
        """
        Gibbs sampler for reversible transiton matrix
        Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)

        Parameters:
        -----------
        T_init : ndarray(n,n)
            An initial transition matrix to seed the sampling. When omitted, the initial transition matrix will
            be constructed from C + C.T, row-normalized. Attention: it is not checked whether T_init is reversible,
            the user needs to ensure this.
        n_step : int
            the number of sampling steps made before returning a new transition matrix. In each sampling step, all
            transition matrix elements are updated.

        """

        for iter in range(n_step):
            for i in range(self.n):
                for j in range(i+1):
                    if self.C[i,j]+self.C[j,i]>0:
                        if i == j:
                            if self._is_positive(self.C[i,i]) and self._is_positive(self.sumC[i]-self.C[i,i]):
                                tmp_t = np.random.beta(self.C[i,i], self.sumC[i]-self.C[i,i])
                                if tmp_t < 1.0:
                                    tmp_x = tmp_t/(1-tmp_t)*(self.X[i,:].sum()-self.X[i,i])
                                    if self._is_positive(tmp_x):
                                        self.X[i,i] = tmp_x
                        else:
                            tmpi = self.X[i,:].sum()-self.X[i,j]
                            tmpj = self.X[j,:].sum()-self.X[j,i]
                            self.X[i,j] = self._update_step(self.X[i,j], tmpi, tmpj, self.C[i,j]+self.C[j,i], self.sumC[i], self.sumC[j])
                            self.X[j,i] = self.X[i,j]
            self.X /= self.X.sum()


    def sample(self, n_step, T_init = None):
        """
        Runs n_step Gibbs sampling steps and returns a new transition matrix. For each step, every element of the
        transition matrix is updated.

        Parameters:
        -----------
        n_step : int
            number of Gibbs sampling steps. Every step samples from the conditional distribution of that element.
        T_init : ndarray (n,n)
            initial transition matrix. If not given, will start from C+C.T, row-normalized

        Returns:
        --------
        The transition matrix after n_step sampling steps

        Raises:
        -------
        ValueError
            if T_init is not a reversible transition matrix

        """
        # T_init given?
        if T_init != None:
            mu = linalg.stationary_distribution(T_init)
            self.X = np.dot(np.diag(mu), T_init)
            # reversible?
            if not np.allclose(self.X, self.X.T):
                raise ValueError('Initial transition matrix is not reversible.')
        # T_init not given and first time calling? Then initialize X
        if self.X is None:
            self.X = self.C + self.C.T
            self.X /= np.sum(self.X)

        # call X-matrix update
        try:
            import tmatrix_sampling as ts
            ts.update(self.C, self.sumC, self.n, self.X, n_step)
        except:
            self._update(n_step)

        T = self.X/self.X.sum(axis=1)[:,None]
        return T


    #TODO: Should be used for efficiency purposes. Currently we just call sample.
    def sample_func(self, eval_fun, n_sample, T_init = None):
        """
        Samples the function of T given.

        eval_fun : python-function
            a function that uses a transition matrix as input
        n_step : int
            number of Gibbs sampling steps. Every step samples from the conditional distribution of that element.
        T_init : ndarray (n,n)
            initial transition matrix. If not given, will start from C+C.T, row-normalized

        Returns:
        --------
        The function value after n_step sampling steps
        """
        T = self.sample(n_sample, T_init = T_init)
        return eval_fun(T)



def main():
    """
    This is a test function

    :return:
    """

    # plot histogram
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from timeit import default_timer as timer

    #C = np.array([[2,1,0],
    #              [1,5,1],
    #              [1,2,10]])
    C = np.array([[787, 54, 27],
                  [60, 2442, 34],
                  [22, 39, 6534]], dtype = np.int32)
    sampler = TransitionMatrixSamplerRev(C)

    t1 = timer()
    nsample = 300000
    nstep   = 1
    x = np.zeros(nsample)
    y = np.zeros(nsample)
    for i in range(nsample):
        P = sampler.sample(nstep)
        #ts.update(C, sumC, n, X, nstep)
        #P = X/X.sum(axis=1)[:,None]
        x[i] = P[0,1]
        y[i] = P[1,0]
    t2 = timer()
    print (t2-t1)


    plt.hist2d(x, y, bins=100, range=((0,1),(0,1)), cmap=cm.jet)
    plt.colorbar()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('sample_c.png')


if __name__ == "__main__":
    main()
