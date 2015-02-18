import math

import numpy as np
cimport numpy as np
import linalg

exp=math.exp
log=math.log
eps=np.spacing(0)


def is_positive(x):
    """
    Helper function, tests if x is numeically positive

    :param x:
    :return:
    """
    return x>=eps and (not math.isinf(x)) and (not math.isnan(x))


def update_step(double v0, double v1, double v2, double c0, double c1, double c2, int random_walk_stepsize=1):
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
    cdef double a = c1+c2-c0
    cdef double b = (c1-c0)*v2+(c2-c0)*v1
    cdef double c = -c0*v1*v2
    cdef double v_bar = 0.5*(-b+(b*b-4*a*c)**.5)/a
    cdef double h = c1/(v_bar + v1)**2 + c2/(v_bar + v2)**2 - c0/v_bar**2
    cdef double k = -h*v_bar*v_bar
    cdef double theta=-1/(h*v_bar)
    cdef double log_prob_new = 0.0
    cdef double log_prob_old = 0.0

    if is_positive(k) and is_positive(theta):
        v0_new = np.random.gamma(k,theta)
        if is_positive(v0_new):
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
    if is_positive(v0_new):
        if v0 == 0:
            v0 = v0_new
        else:
            log_prob_new = c0*log(v0_new)-c1*log(v0_new+v1)-c2*log(v0_new+v2)
            log_prob_old = c0*log(v0)-c1*log(v0+v1)-c2*log(v0+v2)
            if np.random.rand() < exp(min(log_prob_new-log_prob_old,0)):
                v0 = v0_new

    return v0


def update(np.ndarray[np.float64_t, ndim=2] C, np.ndarray[np.float64_t, ndim=1] sumC, int n,
           np.ndarray[np.float64_t, ndim=2] X, int n_step):
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

    cdef int iter,i,j
    cdef double tmp_t,tmp_x,tmpi,tmpj
    for iter in range(n_step):
        for i in range(n):
            for j in range(i+1):
                if C[i,j]+C[j,i]>0:
                    if i == j:
                        if is_positive(C[i,i]) and is_positive(sumC[i]-C[i,i]):
                            tmp_t = np.random.beta(C[i,i], sumC[i]-C[i,i])
                            tmp_x = tmp_t/(1-tmp_t)*(X[i].sum()-X[i,i])
                            if is_positive(tmp_x):
                                X[i,i] = tmp_x
                    else:
                        tmpi = X[i].sum()-X[i,j]
                        tmpj = X[j].sum()-X[j,i]
                        X[i,j] = update_step(X[i,j], tmpi, tmpj, C[i,j]+C[j,i], sumC[i], sumC[j])
                        X[j,i] = X[i,j]
        X /= X.sum()

