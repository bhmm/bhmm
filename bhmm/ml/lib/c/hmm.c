#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __DIMS__
#define __DIMS__
#define DIMM2(arr, i, j)    arr[(i)*M + j]
#define DIM2(arr, i, j)     arr[(i)*N + j]
#define DIM3(arr, t, i , j) arr[(t)*N*N + (i)*N + j]
#define DIMM3(arr, t, i, j) arr[(t)*N*M + (i)*M + j]
#endif

double forward(
        double *alpha,
        double *scaling,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T)
{
    int i, j, t;
    double sum, logprob;

    printf('F1');

    // first alpha and scaling factors
    scaling[0] = 0.0;
    for (i = 0; i < N; i++) {
        alpha[i]  = pi[i] * pobs[i];
        scaling[0] += alpha[i];
    }

    printf('F2');

    // scale first alpha
    if (scaling[0] != 0)
        for (i = 0; i < N; i++)
            alpha[i] /= scaling[0];

    printf('F3');

    // iterate trajectory
    for (t = 0; t < T-1; t++) {
        printf(t);
        scaling[t+1] = 0.0;
        // compute new alpha and scaling
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (i = 0; i < N; i++) {
                sum += alpha[t*N+i]*A[i*N+j];
            }
            alpha[(t+1)*N+j] = sum * pobs[(t+1)*N+j];
            scaling[t+1] += alpha[(t+1)*N+j];
        }
        // scale this row
        if (scaling[t+1] != 0)
            for (j = 0; j < N; j++)
                alpha[(t+1)*N+j] /= scaling[t+1];
    }

    // calculate likelihood
    logprob = 0.0;
    for (t = 0; t < T; t++)
        logprob += log(scaling[t]);
    return logprob;
}


void backward(
        double *beta,
        double *scaling,
        const double *A,
        const double *pobs,
        int N, int T)
{
    int i, j, t;
    double sum;

    // first beta and scaling factors
    scaling[T-1] = 0.0;
    for (i = 0; i < N; i++){
        beta[(T-1)*N+1] = 1.0;
        scaling[T-1] += beta[i];
    }

    // scale first beta
    if (scaling[T-1] != 0)
        for (i = 0; i < N; i++)
            beta[i] /= scaling[T-1];

    // iterate trajectory
    for (t = T-2; t >= 0; t--){
        scaling[t] = 0.0;
        // compute new beta and scaling
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++)
                sum += A[i*N+j]*pobs[t*N+j]*beta[(t+1)*N+j];
            beta[t*N+i] = sum;
            scaling[t] += sum;
        }
        // scale this row
        if (scaling[t+1] != 0)
            for (j = 0; j < N; j++)
                beta[t*N+j] /= scaling[t];
    }
}


void computeGamma(
        double *gamma,
        const double *alpha,
        const double *beta,
        int T, int N)
{
    int i, t;
    double sum;

    for (t = 0; t < T; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++) {
            gamma[t*N+i] = alpha[t*N+i]*beta[t*N+i];
            sum += gamma[t*N+i];
        }
        for (i = 0; i < N; i++)
            gamma[t*N+i] /= sum;
    }
}

void compute_state_counts(
        double *state_counts,
        const double *gamma,
        int T, int N)
{
    int i, t;
    for (i = 0; i < N; i++) {
        state_counts[i] = 0.0;
        for (t = 0; t < T; t++)
            state_counts[i] += gamma[t*N+i];
    }
}


void compute_transition_counts(
        double *transition_counts,
        const double *A,
        const double *pobs,
        const double *alpha,
        const double *beta,
        int N, int T)
{
    int i, j, t;
    double sum, *tmp;
    
    tmp = (double*) malloc(N*N * sizeof(double));
    for (t = 0; t < T-1; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                tmp[i*N+j] = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*pobs[(t+1)*N+i];
                sum += tmp[i*N+j];
            }
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                transition_counts[i*N+j] += tmp[i*N+j] / sum;
    }
    free(tmp);
}


int argmax(double* v, int N)
{
    int i;
    double a = 0;
    double m = v[0];
    for (i = 1; i < N; i++)
        if (v[i] < m){
            a = i;
            m = v[i];
        }
    return a;
}


void compute_viterbi(
        int *path,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T)
{
    int i, j, t, maxj;
    double sum, maxprod, p;

    // allocate v
    double* v = (double*) malloc(N);
    double* vnext = (double*) malloc(N);
    double* h = (double*) malloc(N);
    double* vh;

    // allocate ptr
    int* ptr = (int*) malloc(T*N);

    // initialization of v
    sum = 0.0;
    for (i = 0; i < N; i++)
        v[i] = pobs[i] * pi[i];
        sum += v[i];
    // normalize
    for (i = 0; i < N; i++)
        v[i] /= sum;

    // iteration of v
    for (t = 1; t < T; t++){
        sum = 0.0;
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++)
                h[j] = A[j*N+i] * v[j];
            ptr[t*N + i] = maxj;
            maxj = argmax(h, N);
            vnext[i] = v[maxj];
            sum += vnext[i];
            }
        // normalize
        for (i = 0; i < N; i++)
            vnext[i] /= sum;
        // update v
        vh = v;
        v = vnext;
        vnext = vh;
    }

    // path reconstruction
    path[T-1] = argmax(v,N);
    for (t = T-2; t >= 0; t--){
        path[t] = ptr[(t+1)*N+path[t+1]];
    }
}

/*
void compute_transition_probabilities(
        double *xi,
        const double *A,
        const double *B,
        const short *O,
        const double *alpha,
        const double *beta,
        int N, int M, int T)
{
    int i, j, t;
    double sum;

    for (t = 0; t < T-1; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                DIM3(xi, t, i, j) = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*B[j*M+O[t+1]];
                sum += DIM3(xi, t, i, j);
            }
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                DIM3(xi, t, i, j) /= sum;
            }
    }
}
*/