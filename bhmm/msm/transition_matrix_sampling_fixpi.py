__author__ = 'noe'

import numpy as np
import math

eps=np.spacing(0)

log=math.log
exp=math.exp

def is_positive(x):
    return x>=eps and (not math.isinf(x)) and (not math.isnan(x))

#update the sample x according to
#the distribution x^a1*(s2-x)^a2*(s3-x)^a3
def update_step_2(x,s2,s3,a1,a2,a3,random_walk_stepsize=1):
    if s2>s3:
        s2,a2,s3,a3=s3,a3,s2,a2
    s=s3/s2
    v=x/(s2-x)
    a=a2+1
    b=a2-a1+(a2+a3+1)/(s-1)
    c=s*(a1+1)/(1-s)
    v_bar=0.5*(-b+(b*b-4*a*c)**0.5)/a
    h=-(a1+1)/v_bar**2-a3/(s/(s-1)+v_bar)**2+(a1+a2+a3+2)/(1+v_bar)**2
    k=-h*v_bar*v_bar
    theta=-1/(h*v_bar)
    if is_positive(k) and is_positive(theta):
        v_new=np.random.gamma(k,theta)
        x_new=s2*v_new/(1+v_new)
        if is_positive(v_new) and is_positive(x_new):
            if v==0 or x==0:
                v=v_new
                x=x_new
            else:
                log_prob_new=a1*log(v_new)+a3*log(s/(s-1)+v_new)-(a1+a2+a3+2)*log(1+v_new)
                log_prob_new-=(k-1)*log(v_new)-v_new/theta
                log_prob_old=a1*log(v)+a3*log(s/(s-1)+v)-(a1+a2+a3+2)*log(1+v)
                log_prob_old-=(k-1)*log(v)-v/theta
                if np.random.rand()<exp(min(log_prob_new-log_prob_old,0)):
                    if is_positive(x_new) and is_positive(s2-x_new) and is_positive(s3-x_new):
                        v=v_new
                        x=x_new
    v_new=v*exp(random_walk_stepsize*np.random.randn())
    x_new=s2*v_new/(1+v_new)
    if is_positive(v_new) and is_positive(x_new):
        if v==0 or x==0:
            v=v_new
            x=x_new
        else:
            log_prob_new=(a1+1)*log(v_new)+a3*log(s/(s-1)+v_new)-(a1+a2+a3+2)*log(1+v_new)
            log_prob_old=(a1+1)*log(v)+a3*log(s/(s-1)+v)-(a1+a2+a3+2)*log(1+v)
            if np.random.rand()<exp(min(log_prob_new-log_prob_old,0)):
                if is_positive(x_new) and is_positive(s2-x_new) and is_positive(s3-x_new):
                    v=v_new
                    x=x_new

    return x


# Gibbs sampler for reversible transition matrix with fixed pi
# diag_prior_parameters[i]>-1 is the prior parameters of the i-th diagonal element
# Generally, diag_prior_parameters[i]=0 if the ML estimated T[i,i]>0 and count_matrix[i,i]=0.
#            Otherwise, diag_prior_parameters[i]=a small number-1
# Let T0=ML estimate if it is known.
# return array sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)
def msm_gibbs_sampling_fixed_pi(count_matrix,pii,diag_prior_parameters,sample_num,eval_fun,T0=np.empty(0)):
    n=count_matrix.shape[0]

    if T0.size==0:
        X=count_matrix+((count_matrix+count_matrix.T>0).astype(np.int))*np.random.rand(n,n)*0.1
        T=X/X.sum(1)[:,np.newaxis]
        X=T*pii[:,np.newaxis]
        X=np.minimum(X,X.T)
        X+=np.diag(1.1*pii-X.sum(1))
        X+=X.T
    else:
        X=np.diag(pii).dot(np.maximum(T0,0))

    X/=X.sum()
    T=X/X.sum(1)[:,np.newaxis]
    d=len(eval_fun(T))
    sample_mem=np.empty([sample_num,d])
    for iter in range(sample_num):
        for i in range(n):
            for j in range(i):
                if count_matrix[i,j]+count_matrix[j,i]>0:
                    a1=count_matrix[i,j]+count_matrix[j,i]-1.0
                    a2=count_matrix[i,i]+diag_prior_parameters[i]+0.0
                    a3=count_matrix[j,j]+diag_prior_parameters[j]+0.0
                    s2=X[i,i]+X[i,j]
                    s3=X[j,j]+X[i,j]
                    x_new=update_step_2(X[i,j],s2,s3,a1,a2,a3)
                    xii_new=s2-x_new
                    xjj_new=s3-x_new
                    if min(x_new,xii_new,xjj_new)>0:
                        X[i,i]=xii_new
                        X[j,j]=xjj_new
                        X[i,j]=x_new
                        X[j,i]=x_new

        X/=X.sum()
        T=X/X.sum(1)[:,np.newaxis]
        sample_mem[iter,:]=eval_fun(T)
    return sample_mem
