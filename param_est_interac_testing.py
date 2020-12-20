#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:39:59 2020

@author: ls616
"""

## Interacting Particles SDE (Testing) ##


#################
#### Prelims ####
#################
import sys
sys.path.append("/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/Project Ideas/code")

import numpy as np
from matplotlib import pyplot as plt
from param_est_interac_funcs import *

#################



##################################
#### Test potential functions ####
##################################

## This test checks that the potential functions are being computed correctly

x = np.linspace(0,100,10001)
alpha = np.linspace(-1,-0.1,10)
v = np.zeros((x.shape[0],alpha.shape[0]))
for i in range(alpha.shape[0]):
    v[:,i] = linear_func(x,alpha[i])
    plt.plot(v[:,i])
    
##################################



####################################
#### Test interaction functions ####
####################################

## This test checks that Aij and Lij matrices are being computed correctly

N=4; beta = 0.1

## compute interaction matrix for mean field model
aij_mat = mean_field_mat(N,beta); aij_mat

## compute laplacian matrix
lij_mat = laplacian_mat(N,aij_mat); lij_mat

## reconstruct interaction matrix
aij_mat_2 = interaction_mat(N,lij_mat,4*[beta]); aij_mat_2

## should be zero
np.round(aij_mat - aij_mat_2,10)

####################################




###########################
#### Test sde_sim_func ####
###########################

# This test checks that the mean-field, Aij, Lij implementations are the same

N = 5; T = 500; alpha = 0.1; beta = .1; Aij = mean_field_mat(N,beta)
Lij = laplacian_mat(N,Aij); sigma = 1; x0 = [-10,-5,0,5,10]; dt = 0.1; seed = 1

## Simulations
sim_test1 = sde_sim_func(N=N,T=T,v_func=linear_func,alpha=alpha,Aij=None,
                        Lij = Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)

sim_test2 = sde_sim_func(N=N,T=T,v_func=linear_func,alpha=alpha,Aij=Aij,
                        Lij = None,sigma=sigma,x0=x0,dt=dt,seed=seed)

sim_test3 = sde_sim_func(N=N,T=T,v_func=linear_func,alpha=alpha,Aij=None,
                         Lij=None,beta=beta,sigma=sigma,x0=x0,dt=dt,seed=seed)

## Plot solutions
plt.plot(sim_test1)

## should all overlap
plt.plot(sim_test1[:,1])
plt.plot(sim_test2[:,1])
plt.plot(sim_test3[:,1])

## should be (approx) zero
np.sum(sim_test1 - sim_test2,axis=None)
np.sum(sim_test1 - sim_test3,axis=None)

###########################



#### testing: write this up properly!
#### trying to adapt method in Egilmez to our model

## mean field model with zero potential (= OU process)
beta = 0.75; N = 2
aij_mat = mean_field_mat(N,beta)
lij_mat = laplacian_mat(N,aij_mat)
T = 10; dt = 0.001; v_func = null_func; sigma = 1; x0 = 0; seed = 1
nt = int(np.round(T/dt))
t_vals = np.linspace(0,T,nt+1)

## compute true covariance of process at time t
integrand = np.zeros((nt+1,N,N))
for i in range(nt+1):
    integrand[i,:,:] = sp.linalg.expm(2*lij_mat*(t_vals[i]-T))*dt
integral = np.cumsum(integrand,axis=0)

## sample
true_sample = np.random.multivariate_normal(mean = [0,0], 
                                            cov = integral[nt,:,:], 
                                            size = 100000,)

## plot histogram
plt.hist2d(true_sample[:,0],true_sample[:,1],bins=(100,100),
           range=[[-10,10],[-10,10]])


## compute many (many) samples from model [can take a while to run!]
n_seeds = 100000
empirical_sample = np.zeros((n_seeds,N))
for i in range(n_seeds):
    empirical_sample[i,:] = sde_sim_func(N,T,v_func,alpha,beta=None,Aij=None,
                                         Lij=lij_mat,sigma=1,x0=0,dt=dt,seed=i)[nt,:]

## plot histogram
plt.hist2d(empirical_sample[:,0],empirical_sample[:,1],bins=(100,100),
           range=[[-10,10],[-10,10]])

## good agreement between empirical and true covariance

####