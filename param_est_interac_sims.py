#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:39:42 2020

@author: ls616
"""

## Interacting Particles SDE (Simulations) ##

#################
#### Prelims ####
#################
import numpy as np
import scipy as sp
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import timeit as timeit
import seaborn as sns
import os
import array_to_latex as atl

default_directory = "/Users/ls616"
code_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/McKean-Vlasov Project/code"
results_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/McKean-Vlasov Project/code/results"
fig_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/McKean-Vlasov Project/notes/figures/sde_sims"
#fig_directory = "/Users/ls616/Desktop"

import sys
sys.path.append(code_directory)
from param_est_interac_funcs import * 

#################



#############################
#### Initial Simulations ####
#############################

# Simulate SDE for quadratic potential, mean-field interaction

## Parameters
N = 5; T = 100; alpha = 0.1; beta = .10; Aij = mean_field_mat(N,beta)
Lij = laplacian_mat(N,Aij); sigma = 1; x0 = 10; dt = 0.1; seed = 1

## Simulation
sim_test = sde_sim_func(N=N,T=T,v_func=linear_func,alpha=alpha,Aij=Aij,
                        Lij = Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)

## Plot
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
for i in range(0,N):
    plt.plot(t_vals,sim_test[:,i],linewidth=1)

plt.plot(t_vals,np.mean(sim_test,axis=1),color='black',linewidth=2.0)
plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')

filename = 'sde_sim_1a.pdf'
save_plot(fig_directory,filename)
plt.show()




# Simulate SDE for linear potential, mean-field interaction

## Parameters
N = 5; T = 400; alpha = 0.1; beta = .1; Aij = mean_field_mat(N,beta)
Lij = laplacian_mat(N,Aij); sigma = 1; x0 = [-10,-5,0,5,10]; dt = 0.1; seed = 2

## Simulation
sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,Aij=Aij,
                        Lij = Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)

## Plot
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
for i in range(0,N):
    plt.plot(t_vals,sim_test[:,i],linewidth=1)

plt.plot(t_vals,np.mean(sim_test,axis=1),color='black',linewidth=2.0)
plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
filename = 'sde_sim_1b.pdf'
save_plot(fig_directory,filename)
plt.show()


# Simulate SDE for linear potential, mean-field interaction, different
# values of beta

## Parameters
N = 6; T = 400; alpha = 0;
beta_vals = np.array([0.005,0.01,0.05,0.1,0.2,0.5]);                    
sigma = 0.2; x0 = [-10,-7.5,-5,-2.5,0,2.5]; dt = 0.1; seed = 2
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)

## Simulation
for i in range(beta_vals.shape[0]):
    sim_test = sde_sim_func(N=N,T=T,v_func=linear_func,alpha=alpha,
                            beta=beta_vals[i],Aij = None, Lij = None,
                            sigma=sigma,x0=x0,dt=dt,seed=seed)
    plt.plot(t_vals,sim_test)
    plt.plot(t_vals,np.mean(sim_test,axis=1),color='black',linewidth=4.0)
    plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
    filename = 'sde_sim_2_{}.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()



# Simulate SDE for linear poential, two distinct sets of interacting particles
# , where interaction between 1 (blue) and 2 (orange) strong, and interaction
# between 3 (green) 3 and 4 (red) and 5 (purple) weak

Aij = np.zeros((5,5))
Aij[0,1] = Aij[1,0] = 0.5; Aij[2,3] = Aij[3,2] = 0.1; Aij[3,4] = Aij[4,3] = 0.1

## Parameters
N = 5; T = 250; alpha = 0.1;
Lij = laplacian_mat(N,Aij); sigma = 1; x0 = [5,10,15,20,25]; dt = 0.1; seed = 1

## Simulation
sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,Aij=Aij,
                        Lij = Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)

## Plot
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
for i in range(0,N):
    plt.plot(t_vals,sim_test[:,i],linewidth=1)
plt.plot(t_vals,np.mean(sim_test[:,[0,1]],axis=1),color='black',linewidth=2.0)
plt.plot(t_vals,np.mean(sim_test[:,[2,3,4]],axis=1),color='black',linewidth=2.0)
plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
filename = 'sde_sim_3a.pdf'
save_plot(fig_directory,filename)
plt.show()




# Simulate SDE for linear potential, tri-diagonal interaction matrix

## parameters
N = 5; interaction = 0.02
Aij = tri_diag_interaction_mat(N,interaction); Lij = laplacian_mat(N,Aij)
T = 400; dt = 0.1;
alpha = 0.1; v_func = null_func
sigma = 0.1; x0 = np.linspace(2,10,5);  seed = 2

## Simulation
sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,Aij=Aij,
                        Lij = Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)

## Plot
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
for i in range(0,N):
    plt.plot(t_vals,sim_test[:,i],linewidth=1)
plt.plot(t_vals,np.mean(sim_test,axis=1),color="black",linewidth=2.0)
plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
filename = 'sde_sim_3b.pdf'
save_plot(fig_directory,filename)
plt.show()



    
## Simulate SDE for linear potential, interaction determined by an 
## interaction kernel (gaussian) which is depends on the distance
## between particles

mu = 0.5; sd = 0.5
test_vals = np.linspace(0,1,num=1001)    
test_out = [influence_func_exp(x,mu=mu,sd=sd) for x in test_vals]
plt.plot(test_vals,test_out)

N = 31; T = 50; alpha = 0.1; beta = None; Aij = None; Lij = None; sigma = 0.05; 
x0 = np.linspace(0,3,N)
dt = 0.05; seed = 1; Aij_calc = True; Aij_scale_param = 1

sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                            Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                            seed=seed,Aij_calc=True,
                            Aij_calc_func=Aij_calc_func,
                            Aij_influence_func = influence_func_exp,
                            Aij_scale_param = Aij_scale_param,
                            mu=.2,sd=.2)#1/np.sqrt(2))

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
for j in range(0,N):
    plt.plot(t_vals,sim_test[:,j],linewidth=1)
    plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
#filename = 'sde_sim_4_{}.pdf'.format(i)
#save_plot(fig_directory,filename)
plt.show()



    
## Simulate SDE for linear potential, interaction determined by an 
## interaction kernel (bump function) which depends on the distance
## between particles

centre = [-0.1,0,0.1,0.2,0.3,0.4]
x_vals = np.linspace(0,1.5,201)

## first plot interaction kernel
for i in range(len(centre)):
    y_vals = [influence_func_bump(x,centre[i],width,squeeze) for x in x_vals]
    plt.plot(x_vals,y_vals)
    plt.xlabel(r'$r$'); plt.ylabel(r'${\phi(r)}$')
    filename = 'sde_sim_4_{}.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()

N = 40; T = 40; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.2; 
x0 = np.linspace(0,5,N); dt = 0.1; seed = 2; v_func = null_func; 

Aij_calc = True; Aij_influence_func = influence_func_bump; 
Aij_calc_func = Aij_calc_func; Aij_scale_param = 2 #use 0.8 for row sum normalisation


## in this simulation, we vary the centre of the interaction interval
centre = [-0.1,0,0.1,0.2,0.3,0.4]; width = 1; squeeze = 0.01

for i in range(len(centre)):

    sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                                Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                                seed=seed,Aij_calc=True, 
                                Aij_calc_func = Aij_calc_func,
                                Aij_influence_func = Aij_influence_func,
                                Aij_scale_param = Aij_scale_param,
                                centre = centre[i],width=width,
                                squeeze=squeeze)
    
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    for j in range(0,N):
        plt.plot(t_vals,sim_test[:,j],linewidth=1)
        plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
    filename = 'sde_sim_5_{}.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()
    

## multiple centre parameters
n_kernels = 2
n_centres = 1
centre_true = [0,.5]; width=[1]*n_kernels; squeeze = [.01]*n_kernels
x_vals = np.linspace(0,1.5,201)
y_vals = [influence_func_mult_bump(x,centre_true,width,squeeze) for x in x_vals]
plt.plot(x_vals,y_vals)
plt.xlabel(r'$r$'); plt.ylabel(r'${\phi(r)}$')
filename = 'sde_sim_6.pdf'
save_plot(fig_directory,filename)
plt.show()


# Simulate SDE for linear poential, interaction determined by the influence
# function (= Aij_influence_func) which takes one value (=strength_low) on 
# the interval [0,mid], and one value (=strength_up) on [up,mid] (see 
# Motsch and Tadmor for details)
    
    
## in this simulation, we vary the range of the interaction (i.e., the value
## of 'up'); we assume that the interaction has the same strength on [0,mid] 
## and [mid,up] (i.e. strength_up = strength_low), and keep this value fixed

## parameters
N = 31; T = 20; alpha = 0.1; beta = None; Aij = None; Lij = None; sigma = 0.15; 
x0 = np.linspace(0,3,num=31)
dt = 0.1; seed = 1; Aij_calc = True

## interaction parameters
up = [0.1,0.2,0.3,0.5,0.7,1.0]
up_mid = [0]*len(up)
strength_up = strength_low = 1

Aij_scale_param = 2

## simulate
for i in range(len(up)):

    sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                            Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                            seed=seed,Aij_calc=True, 
                            Aij_calc_func = Aij_calc_func,
                            Aij_influence_func = influence_func_indicator,
                            Aij_scale_param = Aij_scale_param
                            up = up[i],mid=up_mid[i],strength_up= strength_up,
                            strength_low = strength_low)#1/np.sqrt(2))

    ## plot
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    for j in range(0,N):
        plt.plot(t_vals,sim_test[:,j],linewidth=1)
        plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
    filename = 'sde_sim_7_{}.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()




## in this simulation, we vary the ratio between the strength of the
## interaction on [0,mid] and [mid, up] (i.e. the values of strength_low
## and strength_up); we fix the range of the interaction (i.e. the values of
## 'mid' and 'up')

N = 40; T = 30; beta = None; Aij = None; Lij = None; sigma = 0.1; 
x0 = np.random.uniform(0,4,N); dt = 0.1; seed = 1; Aij_calc = True

up = 1
mid = 1/np.sqrt(2)

strength_up = [0.1,1,1,1]
strength_low = [1,1,0.5,0.1]

Aij_scale_param = 2

for i in range(len(strength_up)):

    ## Simulation
    sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                            Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                            seed=seed,Aij_calc=True, 
                            Aij_calc_func = Aij_calc_func,
                            Aij_influence_func = influence_func_indicator,
                            Aij_scale_param = Aij_scale_param,
                            up = up,mid=mid,strength_up = strength_up[i],
                            strength_low = strength_low[i])

    ## Plot
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    for j in range(0,N):
        plt.plot(t_vals,sim_test[:,j],linewidth=1)
        plt.xlabel(r'$t$'); plt.ylabel(r'$x(t)$')
    filename = 'sde_sim_7_{}_1.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()

#############################



########################################
#### Mean Field Estimator (Offline) ####
########################################

# Estimate mean field parameter for different random seeds (MLE and MAP)

# -> performance of MLE and MAP very similar
# -> MAP performs slightly better for small T
# -> MAP and MLE perform equally as well as T increases

N=5;T=500;v_func=linear_func;alpha=0.1;beta=1;sigma=1
x0=2;dt=0.1;seed=1;
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
n_seeds = 20
nt = int(np.round(T/dt))
est_beta = True; est_alpha = True

beta_est_mle = np.zeros((2,nt-1,n_seeds))
for i in range(n_seeds):
    beta_est_mle[:,:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=1,sigma=1,x0=2,dt=0.1,seed=i,
                                  mle=True,est_beta = est_beta, 
                                  est_alpha = est_alpha)


beta_est_map = np.zeros((2,nt-1,n_seeds))
for i in range(n_seeds):
    beta_est_map[:,:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=1,sigma=1,x0=2,dt=0.1,seed=i,
                                  mle=False,est_beta = est_beta,
                                  est_alpha = est_alpha)

## plot MLE estimates (beta)
plt.plot(t_vals[100:],beta_est_mle[1,100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_mle[1,100:],axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MLE}}(t)$');
filename = 'offline_est_sim_1a_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MLE estimates (alpha)
plt.plot(t_vals[100:],beta_est_mle[0,100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_mle[0,100:],axis=1),linewidth=2,color='k');
plt.axhline(y=alpha,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}_{\mathrm{MLE}}(t)$');
filename = 'offline_est_sim_1a_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates (beta)
plt.plot(t_vals[100:],beta_est_map[1,100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_map[1,100:],axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_1b_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates (alpha)
plt.plot(t_vals[100:],beta_est_map[0,100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_map[0,100:],axis=1),linewidth=2,color='k');
plt.axhline(y=alpha,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_1b_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()

# plot l1 errors (beta)
plt.plot(t_vals[100:],abs(beta-beta_est_mle[1,100:,]))
plt.plot(t_vals[100:],np.mean(abs(beta-beta_est_mle[1,100:,]),axis=1),color="black",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}_{\mathrm{MLE}}|$')
filename = 'offline_est_sim_1c_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()


# plot l1 errors (alpha)
plt.plot(t_vals[100:],abs(alpha-beta_est_map[0,100:,]))
plt.plot(t_vals[100:],np.mean(abs(alpha-beta_est_mle[0,100:,]),axis=1),color="black",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\alpha-\hat{\alpha}_{\mathrm{MAP}}|$')
filename = 'offline_est_sim_1d_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


# plot mse (very similar, MAP slightly better for small T) (beta)
plt.plot(t_vals[100:],np.mean((beta-beta_est_mle[1,100:])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MLE}}$")#,color="black",linewidth=2.0)
plt.plot(t_vals[100:],np.mean((beta-beta_est_map[1,100:])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MAP}}$")#,color="grey",linewidth=2.0)
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1e_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

plt.plot(t_vals[0:20],np.mean((beta-beta_est_mle[1,0:20])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MLE}}$")
plt.plot(t_vals[0:20],np.mean((beta-beta_est_map[1,0:20])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MAP}}$")
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1f_beta.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot mse (very similar, MAP slightly better for small T) (alpha)
plt.plot(t_vals[100:],np.mean((alpha-beta_est_mle[0,100:])**2,axis=1),label=r"$\hat{\alpha}_{\mathrm{MLE}}$")#,color="black",linewidth=2.0)
plt.plot(t_vals[100:],np.mean((alpha-beta_est_map[0,100:])**2,axis=1),label=r"$\hat{\alpha}_{\mathrm{MAP}}$")#,color="grey",linewidth=2.0)
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\alpha-\hat{\alpha})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1e_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()

plt.plot(t_vals[0:20],np.mean((alpha-beta_est_mle[0,0:20])**2,axis=1),label=r"$\hat{\alpha}_{\mathrm{MLE}}$")
plt.plot(t_vals[0:20],np.mean((alpha-beta_est_map[0,0:20])**2,axis=1),label=r"$\hat{\alpha}_{\mathrm{MAP}}$")
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\alpha-\hat{\alpha})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1f_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot variance (pretty much identical) (beta)
plt.plot(t_vals[100:],np.var(beta_est_mle[1,100:],axis=1))
#plt.plot(t_vals[100:],np.var(beta_est_map[100:],axis=1))
plt.show()

## plot variance (pretty much identical) (alpha)
plt.plot(t_vals[100:],np.var(beta_est_mle[0,100:],axis=1))
#plt.plot(t_vals[100:],np.var(beta_est_map[100:],axis=1))
plt.show()




# Estimate mean field parameter for different values of beta (MLE and MAP)

# -> estimation successful for all values

N=10;T=200;v_func=linear_func;alpha=0.1;
beta_vals=np.linspace(0,2,5);sigma=1
x0=2;dt=0.1;seed=2;
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_beta = True; est_alpha = True

beta_est_mle = np.zeros((nt-1,beta_vals.shape[0]))
for i in range(beta_vals.shape[0]):
    beta_est_mle[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=beta_vals[i],sigma=1,x0=2,dt=0.1,seed=1,
                                  mle=True,est_beta = est_beta,
                                  est_alpha = est_alpha)[1]


beta_est_map = np.zeros((nt-1,beta_vals.shape[0]))
for i in range(beta_vals.shape[0]):
    beta_est_map[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=beta_vals[i],sigma=1,x0=2,dt=0.1,seed=1,
                                  mle=False,est_beta = est_beta,
                                  est_alpha = est_alpha)[1]
   
## plot MLE estimates
plt.plot(t_vals[20:],beta_est_mle[20:]); 
for i in range(beta_vals.shape[0]): plt.axhline(y=beta_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MLE}}(t)$')
filename = 'offline_est_sim_2a_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates
plt.plot(t_vals[20:],beta_est_map[20:]); 
for i in range(beta_vals.shape[0]): plt.axhline(y=beta_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_2b_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()



# Estimate confinement parameter for different values of alpha (MLE and MAP)

# -> estimation successful for all values (if we increase N and T enough)
# -> this parameter seems to be slightly harder to estimate than the
#    interaction parameter

N=10;T=100;v_func=linear_func
beta = 0.1;sigma=.5
alpha_vals = np.linspace(0,0.5,6)
x0=1;dt=0.1;seed=5;
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_beta = True; est_alpha = True

alpha_est_mle = np.zeros((nt-1,alpha_vals.shape[0]))
for i in range(alpha_vals.shape[0]):
    alpha_est_mle[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,
                                           alpha=alpha_vals[i],beta=beta,
                                           sigma=sigma,x0=2,dt=0.1,seed=seed,
                                           mle=True,est_beta = est_beta,
                                           est_alpha = est_alpha)[0]


alpha_est_map = np.zeros((nt-1,alpha_vals.shape[0]))
for i in range(alpha_vals.shape[0]):
    alpha_est_map[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,
                                           alpha=alpha_vals[i],beta=beta,
                                           sigma=sigma,x0=2,dt=0.1,seed=seed,
                                           mle=False,est_beta = est_beta,
                                           est_alpha = est_alpha)[0]
   
## plot MLE estimates
plt.plot(t_vals[20:],alpha_est_mle[20:]); 
for i in range(alpha_vals.shape[0]): plt.axhline(y=alpha_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}_{\mathrm{MLE}}(t)$')
filename = 'offline_est_sim_2a_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates
plt.plot(t_vals[20:],alpha_est_map[20:]); 
for i in range(alpha_vals.shape[0]): plt.axhline(y=alpha_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_2b_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()




# Estimate mean field parameter for different number of particles (MLE and MAP); 
# trial averaged over several random seeds.

# -> in general, MSE and variance decrease as number of particles increases
# -> the improvement is particularly noticeable for small T
# -> no significant difference between MLE and MAP, especially as T increasees
# -> MAP does perform better for small values of T
# -> NB: this code can take a while to run!

N_vals=np.array([2,5,10,25,50,100]);#np.linspace(10,50,5);
T=30;v_func=linear_func;alpha=0.5;
beta=1;sigma=1; x0=2;dt=0.1;seed_vals=np.linspace(1,500,500);
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_beta = True; est_alpha = True

## compute MLE
beta_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
alpha_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        est_mle = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                     v_func=v_func,alpha=alpha,
                                     beta=beta,sigma=sigma,x0=x0,
                                     dt=dt,seed=np.int(seed_vals[j]),
                                     mle=True,est_beta = est_beta,
                                     est_alpha = est_alpha)
        beta_est_mle[:,i,j] = est_mle[1]
        alpha_est_mle[:,i,j] = est_mle[0]

## compute MAP
if False:
    beta_est_map = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
    alpha_est_map = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            est_map = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                         v_func=v_func,alpha=alpha,
                                         beta=beta,sigma=sigma,x0=x0,
                                         dt=dt,seed=np.int(seed_vals[j]),
                                         mle=False,est_beta = est_beta,
                                         est_alpha = est_alpha)
            beta_est_map[:,i,j] = est_map[1]
            alpha_est_map[:,i,j] = est_map[0]
   
## plot MLE estimates (beta)
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        plt.plot(t_vals[10:],beta_est_mle[10:,i,j])
    plt.plot(t_vals[10:],np.mean(beta_est_mle[10:,i,:],axis=1),linewidth=2,color='k',label="N={}".format(N_vals[i]));
    plt.axhline(y=beta,color='k',linewidth=0.9,linestyle="--")
    plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$')
    plt.legend()
    #filename = 'offline_est_sim_3_{}_beta.pdf'.format(i)
    #save_plot(fig_directory,filename)
    plt.show()
    
## plot MLE estimates (alpha)
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        plt.plot(t_vals[10:],alpha_est_mle[10:,i,j])
    plt.plot(t_vals[10:],np.mean(alpha_est_mle[10:,i,:],axis=1),linewidth=2,color='k',label="N={}".format(N_vals[i]));
    plt.axhline(y=alpha,color='k',linewidth=0.9,linestyle="--")
    plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}(t)$')
    plt.legend()
    #filename = 'offline_est_sim_3_{}_alpha.pdf'.format(i)
    #save_plot(fig_directory,filename)
    plt.show()   
    
  
if False:
    ## plot MAP estimates (beta)
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            plt.plot(t_vals[10:],beta_est_map[10:,i,j])
        plt.plot(t_vals[10:],np.mean(beta_est_map[10:,i,:],axis=1),linewidth=2,color='k');
        plt.axhline(y=beta,color='k',linewidth=0.9,linestyle="--")
        plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$')
        plt.show()
        
    ## plot MAP estimates (alpha)
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            plt.plot(t_vals[10:],alpha_est_map[10:,i,j])
        plt.plot(t_vals[10:],np.mean(alpha_est_map[10:,i,:],axis=1),linewidth=2,color='k');
        plt.axhline(y=alpha,color='k',linewidth=0.9,linestyle="--")
        plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}(t)$')
        plt.show()
    
       
## plot MLE normalised MSE (larger t) (beta)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[1:],np.mean((beta - beta_est_mle[1:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_{2}^{*}-\hat{\theta}_{2}(t))^2\right]$')
    plt.yscale("log")
    plt.legend()  
filename = 'offline_est_sim_3a_beta_log_scale.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MLE normalised MSE (larger t) (alpha)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[1:],np.mean((alpha - alpha_est_mle[1:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_{1}^{*}-\hat{\theta}_{1}(t))^2\right]$')
    plt.legend()  
    plt.yscale("log")
filename = 'offline_est_sim_3a_alpha_log_scale.pdf'
save_plot(fig_directory,filename)
plt.show()
    

if False:
    ## plot MAP normalised MSE (larger t) (beta)
    for i in range(N_vals.shape[0]):
        plt.plot(t_vals[100:],np.mean((beta - beta_est_map[100:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
        plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$')
        plt.legend()
    filename = 'offline_est_sim_3b_beta.pdf'
    #save_plot(fig_directory,filename)
    plt.show()
    
    ## plot MAP normalised MSE (larger t) (alpha)
    for i in range(N_vals.shape[0]):
        plt.plot(t_vals[100:],np.mean((alpha - alpha_est_map[100:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
        plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\alpha-\hat{\alpha}(t))^2\right]$')
        plt.legend()
    filename = 'offline_est_sim_3b_alpha.pdf'
    #save_plot(fig_directory,filename)
    plt.show()


## plot MLE normalised MSE (smaller t) (beta)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((beta - beta_est_mle[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()  
filename = 'offline_est_sim_3c_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MLE normalised MSE (smaller t) (alpha)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((alpha - alpha_est_mle[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()  
filename = 'offline_est_sim_3c_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()
    

## plot MAP normalised MSE (smaller t) (beta)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((beta - beta_est_map[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$')
    plt.legend()
filename = 'offline_est_sim_3d_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MAP normalised MSE (smaller t) (alpha)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((alpha - alpha_est_map[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\alpha-\hat{\alpha}(t))^2\right]$')
    plt.legend()
filename = 'offline_est_sim_3d_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


## plot MLE variance (very similar to MSE) (beta)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(beta_est_mle[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\beta}(t))$')
    plt.legend()
plt.show()


## plot MLE variance (very similar to MSE) (alpha)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(alpha_est_mle[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\alpha}(t))$')
    plt.legend()
plt.show()
    

## plot MAP variance (very similar to MSE) (beta)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(beta_est_map[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\beta}(t))$')
    plt.legend()
plt.show()

## plot MAP variance (very similar to MSE) (alpha)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(alpha_est_map[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\alpha}(t))$')
    plt.legend()
plt.show()




# Compare MSE of MLE as function of N (fixed T)

N_vals=np.linspace(2,400,200);T=5;v_func=linear_func;alpha=0.5;
beta=1;sigma=1; x0=2;dt=0.1;seed_vals=np.linspace(1,500,500);
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_alpha = True; est_beta = True

## compute MLE
beta_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
alpha_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        est_mle = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                               v_func=v_func,alpha=alpha,
                                               beta=beta,sigma=sigma,x0=x0,
                                               dt=dt,seed=np.int(seed_vals[j]),
                                               mle=True,est_beta = est_beta,
                                               est_alpha = est_alpha)
        beta_est_mle[:,i,j] = est_mle[1]
        alpha_est_mle[:,i,j] = est_mle[0]
        
    print(i)

   
## save!
#os.chdir(results_directory)
#np.save("mean_field_offline_beta_l1_error_vs_N_2_to_400_beta_for_paper",beta_est_mle)
#np.save("mean_field_offline_alpha_l1_error_vs_N_2_to_400_alpha_for_paper",alpha_est_mle)
#os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#beta_est_mle = np.load("mean_field_offline_beta_l1_error_vs_N.npy")
#alpha_est_mle = np.load("mean_field_offline_alpha_l1_error_vs_N.npy")
#os.chdir(default_directory)

## plot (beta)
beta_est_l1 = np.mean(abs(beta-beta_est_mle[nt-2]),axis=1)
beta_est_l1_upper = beta_est_l1 + 1.96*np.var(beta_est_mle[nt-2]-beta,axis=1)
beta_est_l1_lower = beta_est_l1 - 1.96*np.var(beta_est_mle[nt-2]-beta,axis=1)
fig1, ax1 = plt.subplots()
ax1.plot(N_vals[10:],beta_est_l1[10:],label=r'$||\hat{\beta}^N_t-\beta_0||$')
ax1.plot(N_vals[10:],.9/np.sqrt(N_vals[10:]),linestyle="--",label=r'$O(N^{-1})$')
#plt.plot(N_vals[2:],beta_est_l1_upper[2:],color="C1",linestyle="--")
#plt.plot(N_vals[2:],beta_est_l1_lower[2:],color="C1",linestyle="--")
ax1.set_xlabel('$N$'); ax1.set_ylabel(r'L1 Error') 
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([20, 50, 100, 200,400])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend()
filename = 'offline_est_sim_7a_beta_N_20_to_400.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot (alpha)
alpha_est_l1 = np.mean(abs(alpha-alpha_est_mle[nt-2]),axis=1)
alpha_est_l1_upper = alpha_est_l1 + 1.96*np.var(alpha_est_mle[nt-2],axis=1)
alpha_est_l1_lower = alpha_est_l1 - 1.96*np.var(alpha_est_mle[nt-2],axis=1)
fig1, ax1 = plt.subplots()
ax1.plot(N_vals[10:],alpha_est_l1[10:],label=r'$||\hat{\alpha}^N_t-\alpha_0||$')
ax1.plot(N_vals[10:],.5/np.sqrt(N_vals[10:]),linestyle="--",label=r'$O(N^{-1})$')
#plt.plot(N_vals[2:],alpha_est_l1_upper[2:],color="C1",linestyle="--")
#plt.plot(N_vals[2:],alpha_est_l1_lower[2:],color="C1",linestyle="--")
ax1.set_xlabel('$N$'); ax1.set_ylabel(r'L1 Error') 
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([20, 50, 100, 200,400])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend()
filename = 'offline_est_sim_7a_alpha_N_20_to_400.pdf'
save_plot(fig_directory,filename)
plt.show()




# Compare MSE of MLE as function of N (fixed T)

N=2;T=2000;v_func=linear_func;alpha=0.5;
beta=1;sigma=1; x0=2;dt=0.1;seed_vals=np.linspace(1,500,500);
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_alpha = True; est_beta = True

## compute MLE
beta_est_mle = np.zeros((nt-1,seed_vals.shape[0]))
alpha_est_mle = np.zeros((nt-1,seed_vals.shape[0]))
for j in range(seed_vals.shape[0]):
    est_mle = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                           beta=beta,sigma=sigma,x0=x0,
                                           dt=dt,seed=np.int(seed_vals[j]),
                                           mle=True,est_beta = est_beta,
                                           est_alpha = est_alpha)
    beta_est_mle[:,j] = est_mle[1]
    alpha_est_mle[:,j] = est_mle[0]
        
    print(j)

   
## save!
os.chdir(results_directory)
np.save("mean_field_offline_beta_l1_error_vs_T_0_to_2000_beta_for_paper",beta_est_mle)
np.save("mean_field_offline_alpha_l1_error_vs_T_0_to_2000_alpha_for_paper",alpha_est_mle)
os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#beta_est_mle = np.load("mean_field_offline_beta_l1_error_vs_N.npy")
#alpha_est_mle = np.load("mean_field_offline_alpha_l1_error_vs_N.npy")
#os.chdir(default_directory)

## plot (beta)
beta_est_l1 = np.mean(abs(beta-beta_est_mle),axis=1)
beta_est_l1_upper = beta_est_l1 + 1.96*np.var(beta_est_mle-beta,axis=1)
beta_est_l1_lower = beta_est_l1 - 1.96*np.var(beta_est_mle-beta,axis=1)
fig1, ax1 = plt.subplots()
start_index = 500
ax1.plot(t_vals[start_index:],beta_est_l1[start_index:],label=r'$||\hat{\beta}^N_t-\beta_0||$')
ax1.plot(t_vals[start_index:],1.3/np.sqrt(t_vals[start_index:]),linestyle="--",label=r'$O(T^{-1})$')
#plt.plot(N_vals[2:],beta_est_l1_upper[2:],color="C1",linestyle="--")
#plt.plot(N_vals[2:],beta_est_l1_lower[2:],color="C1",linestyle="--")
ax1.set_xlabel('$T$'); ax1.set_ylabel(r'L1 Error') 
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([50,100, 200,500,1000,2000])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend()
filename = 'offline_est_sim_8a_beta_T_50_to_2000.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot (alpha)
alpha_est_l1 = np.mean(abs(alpha-alpha_est_mle),axis=1)
alpha_est_l1_upper = alpha_est_l1 + 1.96*np.var(alpha_est_mle-alpha,axis=1)
alpha_est_l1_lower = alpha_est_l1 - 1.96*np.var(alpha_est_mle-alpha,axis=1)
fig1, ax1 = plt.subplots()
start_index = 500
ax1.plot(t_vals[start_index:],alpha_est_l1[start_index:],label=r'$||\hat{\alpha}^N_t-\alpha_0||$')
ax1.plot(t_vals[start_index:],.9/np.sqrt(t_vals[start_index:]),linestyle="--",label=r'$O(T^{-1})$')
#plt.plot(N_vals[2:],beta_est_l1_upper[2:],color="C1",linestyle="--")
#plt.plot(N_vals[2:],beta_est_l1_lower[2:],color="C1",linestyle="--")
ax1.set_xlabel('$T$'); ax1.set_ylabel(r'L1 Error') 
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xticks([50, 100, 200,500,1000,2000])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend()
filename = 'offline_est_sim_8a_alpha_T_50_to_2000.pdf'
save_plot(fig_directory,filename)
plt.show()



## Asymptotic normality ##

N=500;T=2;v_func=linear_func;alpha=0.5;
mu0 = 2; sigma0 = 1
n_samples = 100000
beta=1;sigma=1; seed_vals=np.linspace(1,n_samples,n_samples);
x0 = np.random.normal(mu0,sigma0,N)
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))
est_alpha = True; est_beta = True

## compute MLE
beta_est_mle = np.zeros((nt-1,seed_vals.shape[0]))
alpha_est_mle = np.zeros((nt-1,seed_vals.shape[0]))


for j in range(0,seed_vals.shape[0]):
    est_mle = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                           beta=beta,sigma=sigma,x0=x0,
                                           dt=dt,seed=np.int(seed_vals[j]),
                                           mle=True,est_beta = est_beta,
                                           est_alpha = est_alpha)
    beta_est_mle[:,j] = est_mle[1]
    alpha_est_mle[:,j] = est_mle[0]
        
    print(j)
    

## save!
#os.chdir(results_directory)
#np.save("mean_field_asymptotic_normality_for_paper_beta_N_500_T_2_n_samples_100000",beta_est_mle)
#np.save("mean_field_asymptotic_normality_for_paper_alpha_N_500_T_2_n_samples_100000",alpha_est_mle)
#os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#beta_est_mle = np.load("mean_field_asymptotic_normality_for_paper_beta_N_1000_T_2_n_samples_5000.npy")
#alpha_est_mle = np.load("mean_field_asymptotic_normality_for_paper_alpha_N_1000_T_2_n_samples_5000.npy")
#os.chdir(default_directory)

# plot limits

## compute asymptotic information matrix
mu0 = mu0; sigma0 = sigma0
gamma = -2*(alpha + beta)
T_dash = (nt-2)*dt
ct_theta = 1/gamma**2*(np.exp(gamma*T_dash)-1)-T_dash/gamma + sigma0**2/gamma*(np.exp(gamma*T_dash)-1)
dt_theta = ct_theta - mu0**2/(2*alpha)*(np.exp(-2*alpha*T_dash)-1)
it_theta = np.array([[dt_theta,ct_theta],[ct_theta,ct_theta]])
it_theta_inv = np.linalg.inv(it_theta)



## 1d histograms
top_index = n_samples
plt.hist((alpha_est_mle[nt-2,0:top_index]-alpha),bins=60,density=True,label=r'$\hat{f}(x)$')
mu = 0; sigma = np.sqrt(1/N*it_theta_inv[0,0])
x = np.linspace(min(alpha_est_mle[nt-2,0:top_index]-alpha), max(alpha_est_mle[nt-2,0:top_index]-alpha), 100)
plt.plot(x, sp.stats.norm.pdf(x, mu, sigma),label=r'$f(x)$')
plt.legend()
plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\alpha}_t^N - \alpha)$')
plt.ylabel("Density")
filename = 'offline_est_sim_9a_asymptotic_normal_alpha_N_500_T_2_n_samples_100000.pdf'
#save_plot(fig_directory,filename)
plt.show()


plt.hist((beta_est_mle[nt-2,0:top_index]-beta),bins=60,density=True,label=r'$\hat{f}(x)$')
mu = 0; sigma = np.sqrt(1/N*it_theta_inv[1,1])
x = np.linspace(min(beta_est_mle[nt-2,0:top_index]-beta), max(beta_est_mle[nt-2,0:top_index]-beta), 100)
plt.plot(x, sp.stats.norm.pdf(x, mu, sigma),label=r'$f(x)$')
plt.legend()
plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\beta}_t^N - \beta)$')
plt.ylabel("Density")
filename = 'offline_est_sim_9a_asymptotic_normal_beta_N_500_T_2_n_samples_100000.pdf'
#save_plot(fig_directory,filename)
plt.show()



## 1d histograms (superposition)
top_index = n_samples
plt.hist((alpha_est_mle[nt-2,0:top_index]-alpha),bins=60,density=True,label=r'$\hat{f}(\theta_1)$')
mu = 0; sigma = np.sqrt(1/N*it_theta_inv[0,0])
x = np.linspace(min(alpha_est_mle[nt-2,0:top_index]-alpha), max(alpha_est_mle[nt-2,0:top_index]-alpha), 100)
plt.plot(x, sp.stats.norm.pdf(x, mu, sigma),label=r'$f(\theta_1)$')

plt.hist((beta_est_mle[nt-2,0:top_index]-beta),bins=60,density=True,label=r'$\hat{f}(\theta_2)$')
mu = 0; sigma = np.sqrt(1/N*it_theta_inv[1,1])
x = np.linspace(min(beta_est_mle[nt-2,0:top_index]-beta), max(beta_est_mle[nt-2,0:top_index]-beta), 100)
plt.plot(x, sp.stats.norm.pdf(x, mu, sigma),label=r'$f(\theta_2)$')

plt.legend()
plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\theta}_t^N - \theta)$')
plt.ylabel("Density")
filename = 'offline_est_sim_9a_asymptotic_normal_both_N_500_T_2_n_samples_100000.pdf'
#save_plot(fig_directory,filename)
plt.show()



# 2d histogram

#sns.kdeplot((alpha_est_mle[nt-2]-alpha),(beta_est_mle[nt-2]-beta),shade=True,cbar=True,common_norm=True,cmap=cm.viridis)
#plt.xlim(x_lim[0], x_lim[1])
#plt.ylim(y_lim[0], y_lim[1])
#plt.show()


    


## plot

# setup grid
points = 500
x_lim = np.array([min(alpha_est_mle[nt-2]-alpha), max(alpha_est_mle[nt-2]-alpha)])
y_lim = np.array([min(beta_est_mle[nt-2]-beta), max(beta_est_mle[nt-2]-beta)])
X = np.linspace(x_lim[0], x_lim[1], points)
Y = np.linspace(y_lim[0], y_lim[1], points)
X, Y = np.meshgrid(X, Y)

# mean and variance
mu = np.array([0,0])
Sigma = it_theta_inv

# position array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

## return multivariate guassian on the array pos
## pos is an array with the last dimension containing x_1,x_2,...
def multivariate_gaussian(pos, mu, Sigma):

    # dimension
    n = mu.shape[0]
    
    # sigma 
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    
    # bottom of outside fraction
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    # compute (x-mu)T.Sigma-1.(x-mu) as vector
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    # output
    return np.exp(-fac / 2) / N


# compute distribution
Z = multivariate_gaussian(pos, mu, N**(-1)*Sigma)

# surface plot
if False:
    
    # setup 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    x = alpha_est_mle[nt-2]-alpha
    y = beta_est_mle[nt-2]-beta
    hist, xedges, yedges = np.histogram2d(x, y, bins=(30,30),density=True)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    #rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    # plot
    
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',alpha=0.1)#,color=rgba)
    
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1,cmap=cm.viridis)
    
    ax.view_init(30, 90)
    
    plt.show()

    # limits and view angle
    #ax.set_zlim(-0.15,2)
    #ax.set_zticks(np.linspace(0,2,5))
    #ax.set_xticks(np.linspace(x_lim[0], x_lim[1],7))
    #ax.set_yticks(np.linspace(y_lim[0], y_lim[1],7))
    


# contour plot
if True:
    
    # plot kde plot
    fig = plt.figure()
    ax = plt.subplot()
    sns.kdeplot((alpha_est_mle[nt-2]-alpha),(beta_est_mle[nt-2]-beta),
                shade=True,cmap = cm.viridis,cbar=True,vmin=0,
                cbar_kws={"ticks":[0,20,40,60,80,100,120]},
                shade_lowest=True,n_levels=100)
    plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\alpha}_t^N - \alpha)$')
    plt.ylabel(r'$N^{-\frac{1}{2}}(\hat{\beta}_t^N - \beta)$')
    plt.xlim(np.min(X),np.max(X))
    plt.ylim(np.min(Y),np.max(Y))
    plt.show()
    
    fig = plt.figure()
    ax = plt.subplot()
    h,x,y,p = ax.hist2d((alpha_est_mle[nt-2]-alpha),(beta_est_mle[nt-2]-beta),bins=(50,50),density=True,vmin=0,vmax=140)
    #plt.clf()
    #plt.close()
    #fig = plt.figure(figsize=[6, 4])
    #ax = plt.subplot()
    #ax.imshow(h,interpolation="gaussian",origin="lower",
    #          extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)], aspect=4.05/5*(np.max(X)-np.min(X))/(np.max(Y)-np.min(Y)))
    fig.colorbar(p,ax=ax,ticks=[0,20,40,60,80,100,120,140])
    plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\theta_1}_t^N - \theta_1)$')
    plt.ylabel(r'$N^{-\frac{1}{2}}(\hat{\theta_2}_t^N - \theta_2)$')
    filename = 'offline_est_sim_9a_asymptotic_normal_bivariate_N_500_T_2_n_samples_100000_hist.pdf'
    #save_plot(fig_directory,filename)
    plt.show()

    # plot contour
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot()
    levels = np.linspace(0, 140, 100)
    h = ax.contourf(X,Y,Z,levels=levels,cmap = cm.viridis,vmax=140)#,vmin=0,vmax=60) ##use vmax=126.5 to compare to kde plot
    fig.colorbar(h,ax=ax,ticks=[0,20,40,60,80,100,120,140]) 
    plt.xlabel(r'$N^{-\frac{1}{2}}(\hat{\theta_1}_t^N - \theta_1)$')
    plt.ylabel(r'$N^{-\frac{1}{2}}(\hat{\theta_2}_t^N - \theta_2)$')
    filename = 'offline_est_sim_9a_asymptotic_normal_bivariate_N_500_T_2_n_samples_100000_true.pdf'
    #save_plot(fig_directory,filename)
    plt.show()
    
    # limits and labels
    #ax.set_xticks(np.linspace(x_lim[0], x_lim[1],5))
    #ax.set_yticks(np.linspace(y_lim[0], y_lim[1],5))

plt.show()

########################################




#######################################
#### Mean Field Estimator (Online) ####
#######################################

# estimate mean field parameter & average over different random seeds
# w. decreasing step size

N = 10; T = 1000; v_func = linear_func; alpha = 0.1; beta = 1;
sigma = 1; x0 = 2; dt = 0.1; est_beta = True; est_alpha = True

n_seeds = 20

np.random.seed(1)
beta0 = np.random.uniform(-1,3,n_seeds); # randomly sample beta0
alpha0 = np.random.uniform(0,2,n_seeds); # randomly sample alpha0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = int(np.round(T/dt))

## step sizes
step_size = np.zeros(nt)
step_size_init = 0.1
for i in range(nt):
    step_size[i] = min(step_size_init,step_size_init/(t_vals[i]**0.6))

## RML estimate
beta_est_rml = np.zeros((nt+1,n_seeds))
alpha_est_rml = np.zeros((nt+1,n_seeds))
for i in range(n_seeds):
    est = mean_field_rmle(N=N,T=T,v_func=v_func,alpha=alpha,
                                        beta=beta,beta0=beta0[i],sigma=sigma,
                                        x0=x0,dt=dt,seed=i,
                                        step_size=step_size,
                                        est_beta = est_beta,
                                        est_alpha = est_alpha,
                                        alpha0 = alpha0[i])
    beta_est_rml[:,i] = est[0]
    alpha_est_rml[:,i] = est[1]


## plot MLE estimates (beta)
plt.plot(t_vals[100:],beta_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(beta_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$');
filename = 'online_est_sim_1a_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MLE estimates (alpha)
plt.plot(t_vals[100:],alpha_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(alpha_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=alpha,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}(t)$');
filename = 'online_est_sim_1a_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


## plot

## plot L1 error (beta)
plt.plot(t_vals[200:],abs(beta-beta_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(beta-beta_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\alpha-\hat{\alpha}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}(t)|$');
plt.legend()
plt.yscale("log")
filename = 'online_est_sim_1b_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot L1 error (alpha)
plt.plot(t_vals[200:],abs(alpha-alpha_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(alpha-alpha_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\alpha-\hat{\alpha}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\alpha-\hat{\alpha}(t)|$');
plt.legend()
filename = 'online_est_sim_1b_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


## plot MSE 
plt.plot(t_vals[200:],(beta-beta_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((beta-beta_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\beta-\hat{\beta}(t))^2$');
plt.legend()
filename = 'online_est_sim_1c_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MSE 
plt.plot(t_vals[200:],(alpha-alpha_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((alpha-alpha_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\alpha-\hat{\alpha}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\alpha-\hat{\alpha}(t))^2$');
plt.legend()
filename = 'online_est_sim_1c_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()




# estimate mean field parameter & average over different random seeds
# w. constannt step size
N = 10; T = 1000; v_func = linear_func; alpha = 0.1; beta = 1;
sigma = 1; x0 = 2; dt = 0.1; est_beta = True; est_alpha = True

n_seeds = 20

np.random.seed(1)
beta0 = np.random.uniform(-1,3,n_seeds); # randomly sample beta0
alpha0 = np.random.uniform(0,2,n_seeds); # randomly sample alpha0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = int(np.round(T/dt))

## step sizes
step_size = 0.01

## RML estimate
beta_est_rml = np.zeros((nt+1,n_seeds))
alpha_est_rml = np.zeros((nt+1,n_seeds))
for i in range(n_seeds):
    est = mean_field_rmle(N=N,T=T,v_func=v_func,alpha=alpha,
                                        beta=beta,beta0=beta0[i],sigma=sigma,
                                        x0=x0,dt=dt,seed=i,
                                        step_size=step_size,
                                        est_beta = est_beta,
                                        est_alpha = est_alpha,
                                        alpha0 = alpha0[i])
    beta_est_rml[:,i] = est[0]
    alpha_est_rml[:,i] = est[1]


## plot MLE estimates (beta)
plt.plot(t_vals[100:],beta_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(beta_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$');
filename = 'online_est_sim_1d_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MLE estimates (alpha)
plt.plot(t_vals[100:],alpha_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(alpha_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=alpha,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}(t)$');
filename = 'online_est_sim_1d_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


## plot

## plot L1 error (beta)
plt.plot(t_vals[200:],abs(beta-beta_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(beta-beta_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\alpha-\hat{\alpha}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}(t)|$');
plt.legend()
filename = 'online_est_sim_1d_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot L1 error (alpha)
plt.plot(t_vals[200:],abs(alpha-alpha_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(alpha-alpha_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\alpha-\hat{\alpha}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\alpha-\hat{\alpha}(t)|$');
plt.legend()
filename = 'online_est_sim_1d_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()


## plot MSE 
plt.plot(t_vals[200:],(beta-beta_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((beta-beta_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\beta-\hat{\beta}(t))^2$');
plt.legend()
filename = 'online_est_sim_1d_beta.pdf'
#save_plot(fig_directory,filename)
plt.show()

## plot MSE 
plt.plot(t_vals[200:],(alpha-alpha_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((alpha-alpha_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\alpha-\hat{\alpha}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\alpha-\hat{\alpha}(t))^2$');
plt.legend()
filename = 'online_est_sim_1d_alpha.pdf'
#save_plot(fig_directory,filename)
plt.show()




# Estimate mean field parameter for different number of particles; and average
# over different random seeds.

N_vals=np.array([2,5,10,25,50,100])#np.array([2,5,10,25,50,100])#np.linspace(2,25,5);
T=1000; v_func=linear_func; alpha=.5; beta = .1;
n_runs = 500
sigma=1; dt=0.1; x0=1; seed_vals=np.linspace(1,n_runs,n_runs);
norm = True; 

est_both_params = False
est_solo_params = True
    
if est_solo_params:
    start_loop = 0
else:
    start_loop = 2
    
if est_both_params:
    end_loop = 3
else:
    end_loop = 2
    

for m in range(start_loop,end_loop):
    if m==0:
        est_beta = True; est_alpha = False
    
    if m==1:
        est_beta = False; est_alpha = True
        
    if m==2: 
        est_beta = True; est_alpha = True

    np.random.seed(1)

    ## for joint estimation
    if m==2:
        beta0 = np.random.uniform(-2,2,n_runs); # randomly sample beta0
        alpha0 = np.random.uniform(-1,2,n_runs); # randomly sample alpha0
    
    
    ## for solo estimation
    if (m==0 or m==1):
        beta0 = np.random.uniform(2,5,n_runs); # randomly sample beta0
        alpha0 = np.random.uniform(2,5,n_runs); # randomly sample alpha0
    
    
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    nt = np.int(round(T/dt))
    
    ## step sizes
    step_size = np.zeros((nt,2))
    
    # for joint estimation
    if m==2:
        step_size_init = [0.1,0.25] 
        step_size = step_size_init
    
    # for solo estimation
    if (m==0 or m==1):
        
        step_size_init = [0.05,0.3] 
        
        for i in range(nt):
            step_size[i,0] = min(step_size_init[0],step_size_init[0]/(t_vals[i]**0.51))
            step_size[i,1] = min(step_size_init[1],step_size_init[1]/(t_vals[i]**0.51))
    
    ## RML estimate (IPS)
    if m==0:
        beta_est_rml_ips = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        beta_est_rml_mvsde = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        
    if m==1:
        alpha_est_rml_ips = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        alpha_est_rml_mvsde = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
    
    if m==2:
        beta_est_rml_ips_both = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        beta_est_rml_mvsde_both = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        alpha_est_rml_ips_both = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
        alpha_est_rml_mvsde_both = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
    
    
    
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            
            if True:
                est_ips = mean_field_rmle(N=np.int(N_vals[i]),T=T,v_func=v_func,alpha=alpha,
                                                beta=beta,beta0=beta0[j],sigma=sigma,
                                                x0=x0,dt=dt,seed=np.int(seed_vals[j]),
                                                step_size=step_size,
                                                est_beta = est_beta,
                                                est_alpha = est_alpha,
                                                alpha0 = alpha0[j],
                                                norm = norm)
                
                if m==0:
                    beta_est_rml_ips[:,i,j] = est_ips[0]
                
                if m==1:
                    alpha_est_rml_ips[:,i,j] = est_ips[1]
                    
                if m==2:
                    beta_est_rml_ips_both[:,i,j] = est_ips[0]
                    alpha_est_rml_ips_both[:,i,j] = est_ips[1]
            
            if True:
                est_mvsde = mvsde_rmle(N=np.int(N_vals[i]),T=T,alpha=alpha,beta=beta,
                                  alpha0 = alpha0[j], beta0=beta0[j],sigma=sigma,
                                  x0=x0,dt=dt,seed=np.int(seed_vals[j]),
                                  step_size=step_size,est_beta = est_beta,
                                  est_alpha = est_alpha, norm = norm,
                                  sim_mvsde = False, sim_particle = True)
                
                if m==0:
                    beta_est_rml_mvsde[:,i,j] = est_mvsde[0]
                    
                if m==1:
                    alpha_est_rml_mvsde[:,i,j] = est_mvsde[1]
                    
                if m==2:
                    beta_est_rml_ips_both[:,i,j] = est_ips[0]
                    alpha_est_rml_ips_both[:,i,j] = est_ips[1]
        
            print(i,j)
        
    ## save!
    os.chdir(results_directory)
    
    if m==0:
        np.save("mean_field_rmle_beta_different_N_T_1000_n_samples_500_ips_v4",beta_est_rml_ips)
        np.save("mean_field_rmle_beta_different_N_T_1000_n_samples_500_mvsde_v4",beta_est_rml_mvsde)
    
    if m==1: 
        np.save("mean_field_rmle_alpha_different_N_T_1000_n_samples_500_ips_v4",alpha_est_rml_ips)
        np.save("mean_field_rmle_alpha_different_N_T_1000_n_samples_500_mvsde_v4",alpha_est_rml_mvsde)
        
    if m==2:
        np.save("mean_field_rmle_alpha_different_N_T_1000_n_samples_500_both_params_ips_v4",alpha_est_rml_ips_both)
        np.save("mean_field_rmle_alpha_different_N_T_1000_n_samples_500_both_params_mvsde_v4",alpha_est_rml_mvsde_both)
        np.save("mean_field_rmle_beta_different_N_T_1000_n_samples_500_both_params_ips_v4",alpha_est_rml_ips_both)
        np.save("mean_field_rmle_beta_different_N_T_1000_n_samples_500_both_params_mvsde_v4",alpha_est_rml_mvsde_both)
    
    os.chdir(default_directory)
    
    
    
## reopen!
#os.chdir(results_directory)
#beta_est_rml = np.load("mean_field_rmle_beta_different_N_T_1000_n_samples_500.npy")
#alpha_est_rml = np.load("mean_field_rmle_alpha_different_N_T_1000_n_samples_500.npy")
#os.chdir(default_directory)    
   

if False:
    ## plot estimates (beta)
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            plt.plot(t_vals[:],beta_est_rml_ips[:,i,j])
        plt.plot(t_vals[:],np.mean(beta_est_rml_ips[:,i,:],axis=1),linewidth=2,color='k');
        plt.axhline(y=beta,color='k',linewidth=0.9,linestyle="--")
        plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$')
        #plt.ylim(0,4)
        filename = 'online_est_sim_2a_beta_both_param_est_{}_v4.pdf'.format(i)
        #save_plot(fig_directory,filename)
        plt.show()
        
    ## plot estimates (alpha)
    for i in range(N_vals.shape[0]):
        for j in range(seed_vals.shape[0]):
            plt.plot(t_vals,alpha_est_rml_ips[:,i,j])
        plt.plot(t_vals,np.mean(alpha_est_rml_ips[:,i,:],axis=1),linewidth=2,color='k');
        plt.axhline(y=alpha,color='k',linewidth=0.9,linestyle="--")
        plt.xlabel('t'); plt.ylabel(r'$\hat{\alpha}(t)$')
        filename = 'online_est_sim_2a_alpha_both_param_est_{}_v4.pdf'.format(i)
        #save_plot(fig_directory,filename)
        plt.show()

       
## plot MSE (beta) (IPS)
t_start_index = 0
for i in range(N_vals.shape[0]):
    plt.plot((t_vals[t_start_index:]),np.mean((beta - beta_est_rml_ips[t_start_index:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_2-\hat{\theta}_{2,t})^2\right]$')
    #plt.plot(t_vals[0:],(t_vals[0:])**(-0.7))
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_beta_ips_mse_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MSE (beta) (MVSDE)
t_start_index = 0
for i in range(N_vals.shape[0]):
    plt.plot((t_vals[t_start_index:]),np.mean((beta - beta_est_rml_mvsde[t_start_index:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_2-\hat{\theta}_{2,t})^2\right]$')
    #plt.plot(t_vals[0:],(t_vals[0:])**(-0.7))
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_beta_mvsde_mse_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MSE (alpha) (IPS)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[t_start_index:],np.mean((alpha - alpha_est_rml_ips[t_start_index:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_1-\hat{\theta}_{1,t})^2\right]$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_alpha_ips_mse_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MSE (alpha) (MVSDE)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[t_start_index:],np.mean((alpha - alpha_est_rml_mvsde[t_start_index:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta_1-\hat{\theta}_{1,t})^2\right]$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_alpha_mvsde_mse_v4.pdf'
save_plot(fig_directory,filename)
plt.show()


## plot variance (beta) (IPS)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.var(beta_est_rml_ips[:,i,:],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\theta_{2,t})$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_beta_ips_variance_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot variance (beta) (MVSDE)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.var(beta_est_rml_mvsde[:,i,:],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\theta_{2,t})$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_beta_mvsde_variance_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot variance (alpha) (IPS)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.var(alpha_est_rml_ips[:,i,:],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\theta_{1,t})$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_alpha_ips_variance_v4.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot variance (alpha) (MVSDE)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.var(alpha_est_rml_mvsde[:,i,:],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\theta_{1,t})$')
    #plt.yscale("log")
    plt.xscale("log")
    plt.legend()
filename = 'mean_field_rmle_T_5000_n_samples_500_alpha_mvsde_variance_v4.pdf'
save_plot(fig_directory,filename)
plt.show()


## compare mse of mvsde and ips estimator
t_start_index = 500
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[t_start_index:],100*abs(np.mean(beta_est_rml_ips[t_start_index:,i,:],axis=1)-beta),label="IPS" %N_vals[i],color="C%d" %i)
    plt.plot(t_vals[t_start_index:],100*abs(np.mean(beta_est_rml_mvsde[t_start_index:,i,:],axis=1)-beta),label="MVSDE" %N_vals[i],linestyle="--",color="C%d" %i)
    plt.xscale("log")
    plt.xlabel("t"); plt.ylabel("Percentage Error")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    filename = "mean_field_rmle_T_5000_n_sample_500_beta_mvsde_ips_mse_comparison_v4_N_{}.pdf".format(N_vals[i])
    save_plot(fig_directory,filename)
    plt.show()
    
#######################################





#################################
#### Lij Estimator (Offline) ####
#################################

## estimate mean field matrix ##
N=5;T=1000;v_func=linear_func;alpha=0.01;
beta=0.5;sigma=1; x0=2;dt=0.01;seed = 1
aij_mat = mean_field_mat(N,beta)
lij_mat = laplacian_mat(N,aij_mat)

ridge_params = [0,10**6]
mle_ridge = True
lasso = False
lasso_param = 0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

# compute mle & ridge
lij_mle = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_params[0],mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat)

lij_ridge = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_params[1],mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat)  

# plot matrix (true)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mat)
axes.set_xticklabels(np.linspace(0,8,9).astype(int))
axes.set_yticklabels(np.linspace(0,8,9).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_4a.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot matrix (mle)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mle[lij_mle.shape[0]-1])
axes.set_xticklabels(np.linspace(0,8,9).astype(int))
axes.set_yticklabels(np.linspace(0,8,9).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_4b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot matrix (ridge)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mle[lij_ridge.shape[0]-1])
axes.set_xticklabels(np.linspace(0,8,9).astype(int))
axes.set_yticklabels(np.linspace(0,8,9).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_4c.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot all mle estimates
for i in range(N):
    for j in range(N):
            plt.plot(t_vals[1000:],lij_mle[1000:,i,j])
plt.axhline(y = lij_mat[0,0],color="k",linestyle="--",linewidth=2)
plt.axhline(y = lij_mat[0,1],color="k",linestyle="--",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$\hat{L}^{ij}_{\mathrm{MLE}}(t)$')
filename = 'offline_est_sim_4d.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot all ridge estimates
for i in range(N):
    for j in range(N):
            plt.plot(t_vals[1000:],lij_ridge[1000:,i,j])
plt.axhline(y = lij_mat[0,0],color="k",linestyle="--",linewidth=2)
plt.axhline(y = lij_mat[0,1],color="k",linestyle="--",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$\hat{L}^{ij}_{\mathrm{RIDGE}}(t)$')
filename = 'offline_est_sim_4e.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot L1 error norm (absolute & relative) (MLE & RIDGE) [small t]
lij_mle_l1_norm_abs = np.sum(abs(lij_mle-lij_mat),axis=(1,2))
lij_mle_l1_norm_rel = np.sum(abs(lij_mle-lij_mat),axis=(1,2))/np.sum(abs(lij_mat))
plt.plot(t_vals[100:1000],lij_mle_l1_norm_abs[100:1000],label="MLE (Absolute)",color="C0")
plt.plot(t_vals[100:1000],lij_mle_l1_norm_rel[100:1000],label="MLE (Relative)",color="C0",linestyle="--")

lij_ridge_l1_norm_abs = np.sum(abs(lij_ridge-lij_mat),axis=(1,2))
lij_ridge_l1_norm_rel = np.sum(abs(lij_ridge-lij_mat),axis=(1,2))/np.sum(abs(lij_mat))
plt.plot(t_vals[100:1000],lij_ridge_l1_norm_abs[100:1000],label="Ridge (Absolute)",color="C1")
plt.plot(t_vals[100:1000],lij_ridge_l1_norm_rel[100:1000],label="Ridge (Relative)",color="C1",linestyle="--")

plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_4f.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot L1 error norm (absolute & relative) (MLE & RIDGE) [large t]
lij_mle_l1_norm_abs = np.sum(abs(lij_mle-lij_mat),axis=(1,2))
lij_mle_l1_norm_rel = np.sum(abs(lij_mle-lij_mat),axis=(1,2))/np.sum(abs(lij_mat))
plt.plot(t_vals[10000:],lij_mle_l1_norm_abs[10000:],label="MLE (Absolute)",color="C0")
plt.plot(t_vals[10000:],lij_mle_l1_norm_rel[10000:],label="MLE (Relative)",color="C0",linestyle="--")

lij_ridge_l1_norm_abs = np.sum(abs(lij_ridge-lij_mat),axis=(1,2))
lij_ridge_l1_norm_rel = np.sum(abs(lij_ridge-lij_mat),axis=(1,2))/np.sum(abs(lij_mat))
plt.plot(t_vals[10000:],lij_ridge_l1_norm_abs[10000:],label="Ridge (Absolute)",color="C1")
plt.plot(t_vals[10000:],lij_ridge_l1_norm_rel[10000:],label="Ridge (Relative)",color="C1",linestyle="--")

plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_4g.pdf'
save_plot(fig_directory,filename)
plt.show()


# reconstruct aij_mat mle
aij_mle = np.zeros((nt-1,N,N))
for i in range(nt-1):
    colsums = mean_field_col_sums(N,lij_mle[i])
    aij_mle[i,:,:] = interaction_mat(N,lij_mle[i],colsums)

# reconstruct beta mle (mean of independent terms in Aij)
beta_mle = np.zeros(nt-1)
for i in range(nt-1):
    beta_mle[i] = N*np.mean(upper_tri_indexing(aij_mle[i,:,:]))

# reconstruct beta mle (mean of all terms in Aij <-> equivalent to above)
#beta_mle = N*np.mean(aij_mle,axis=(1,2))

# compute beta mle (direct)
beta_mle_direct = mean_field_mle_map(N,T,v_func,alpha,beta,sigma,x0,dt,seed,mle=True)[0]


# plot
plt.plot(t_vals[1000:],beta_mle[1000:],label="Indirect"); 
plt.plot(t_vals[1000:],beta_mle_direct[1000:],label="Direct"); 
plt.axhline(y=beta,color='k',linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\beta(t)$')
plt.legend()
filename = 'offline_est_sim_4h.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot mse (indirect vs direct)
plt.plot(t_vals[10000:],(beta_mle[10000:]-beta)**2,label = "Indirect")
plt.plot(t_vals[10000:],(beta_mle_direct[10000:]-beta)**2, label="Direct")
plt.xlabel('t'); plt.ylabel(r'$L_2$ Error')
plt.legend()
filename = 'offline_est_sim_4i.pdf'
save_plot(fig_directory,filename)
plt.show()





## estimate tridiagonal matrix using mle, ridge, lasso ##
N = 10; T = 100; v_func = linear_func; alpha = 0.1;
sigma = 0.1; x0 = 2; dt = 0.1; seed = 1

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

interaction = 0.1
aij_mat = tri_diag_interaction_mat(N,interaction)
lij_mat = laplacian_mat(N,aij_mat)

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))



# compute mle
ridge_param = 0; mle_ridge = True; lasso = False; lasso_param = 0
lij_mle = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=None,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat) 

# compute ridge
ridge_param = 10^3; mle_ridge = True; lasso = False; lasso_param = 0
lij_ridge = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=None,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat) 

# compute lasso
lij_mat_init = laplacian_mat(N,aij_mat)-0.2
ridge_param = 0; mle_ridge = False; lasso = True; lasso_param = 0.01; lasso_t_vals=np.arange(10,110,10)
lij_lasso = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=None,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat_init,lasso_t_vals=lasso_t_vals) 

# plot true matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mat,interpolation="nearest")
axes.set_xticks(np.arange(0,10,1))
axes.set_yticks(np.arange(0,10,1))
axes.set_xticklabels(np.linspace(1,10,10).astype(int))
axes.set_yticklabels(np.linspace(1,10,10).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_5a.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot mle matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mle[nt-2],interpolation="nearest")
axes.set_xticks(np.arange(0,10,1))
axes.set_yticks(np.arange(0,10,1))
axes.set_xticklabels(np.linspace(1,10,10).astype(int))
axes.set_yticklabels(np.linspace(1,10,10).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_5b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot ridge matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_ridge[nt-2],interpolation="nearest")
axes.set_xticks(np.arange(0,10,1))
axes.set_yticks(np.arange(0,10,1))
axes.set_xticklabels(np.linspace(1,10,10).astype(int))
axes.set_yticklabels(np.linspace(1,10,10).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_5c.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot lasso matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_lasso[len(lasso_t_vals)-1],interpolation="nearest")
axes.set_xticks(np.arange(0,10,1))
axes.set_yticks(np.arange(0,10,1))
axes.set_xticklabels(np.linspace(1,10,10).astype(int))
axes.set_yticklabels(np.linspace(1,10,10).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_5d.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot l1 norm (relative)
mle_l1_rel= np.zeros(nt-2)
ridge_l1_rel= np.zeros(nt-2)
mle_l1_abs= np.zeros(nt-2)
ridge_l1_abs= np.zeros(nt-2)
for i in range(nt-2):
    mle_l1_rel[i] = l1_norm(lij_mle[i]-lij_mat)/l1_norm(lij_mat)
    ridge_l1_rel[i] = l1_norm(lij_ridge[i]-lij_mat)/l1_norm(lij_mat)
    mle_l1_abs[i] = l1_norm(lij_mle[i]-lij_mat)
    ridge_l1_abs[i] = l1_norm(lij_ridge[i]-lij_mat)

lasso_l1_rel = np.zeros(len(lasso_t_vals))
lasso_l1_abs = np.zeros(len(lasso_t_vals))
for i in range(len(lasso_t_vals)):
    lasso_l1_rel[i] = l1_norm(lij_lasso[i]-lij_mat)/l1_norm(lij_mat)
    lasso_l1_abs[i] = l1_norm(lij_lasso[i]-lij_mat)


# plot
plt.plot(t_vals[98:(nt-2)],mle_l1_rel[98:],label="MLE"); 
plt.plot(t_vals[98:(nt-2)],ridge_l1[98:],label="RIDGE"); 
plt.plot(np.round(lasso_t_vals),lasso_l1_rel,label="LASSO")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_5e.pdf'
save_plot(fig_directory,filename)
plt.show()

plt.plot(t_vals[98:(nt-2)],mle_l1_abs[98:],label="MLE"); 
plt.plot(t_vals[98:(nt-2)],ridge_l1_abs[98:],label="RIDGE"); 
plt.plot(np.round(lasso_t_vals),lasso_l1_abs,label="LASSO")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_5f.pdf'
save_plot(fig_directory,filename)
plt.show()




## estimate sparse aij matrix using mle, ridge, lasso ##
N = 8; T=50;v_func=linear_func;alpha=0.1;
sigma=1; x0=2;dt=0.1;seed = 1

aij_mat = np.diag((1,1,1,1,12,2,2,2)) + np.diag((0.5,0.5,0.5,0.5,0.3,0.3,0.4),1) + np.diag((0.5,0.5,0.5,0.5,0.3,0.3,0.4),-1) +np.diag((0.4,0.4),6) + np.diag((0.4,0.4),-6)
aij_mat[0,3] = aij_mat[3,0] = 0.9
lij_mat = laplacian_mat(N,aij_mat)


t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

# compute mle
ridge_param = 0; mle_ridge = True; lasso = False; lasso_param = 0
lij_mle = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat) 


# compute ridge 
ridge_param = 10^6; mle_ridge = True; lasso = False; lasso_param = 0
lij_ridge = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat) 


# compute lasso 
ridge_param = 0; mle_ridge = False; lasso = True; lasso_param = 0.005; lasso_t_vals=np.arange(5,55,5)
lij_lasso = Lij_mle_ridge_lasso(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                              Lij = lij_mat,sigma=sigma,x0=x0,dt=dt,seed=seed,
                              ridge_param=ridge_param,mle_ridge=mle_ridge,
                              lasso=lasso,lasso_param = lasso_param,
                              Lij_init=lij_mat,lasso_t_vals=lasso_t_vals) 


# plot true matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mat,interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_6a.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot mle matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mle[nt-2],interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_6b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot ridge matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_ridge[nt-2],interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_6c.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot lasso matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_lasso[len(lasso_t_vals)-1],interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'offline_est_sim_6d.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot l1 norm (relative)
mle_l1_rel= np.zeros(nt-2)
ridge_l1_rel= np.zeros(nt-2)
mle_l1_abs= np.zeros(nt-2)
ridge_l1_abs= np.zeros(nt-2)
for i in range(nt-2):
    mle_l1_rel[i] = l1_norm(lij_mle[i]-lij_mat)/l1_norm(lij_mat)
    ridge_l1_rel[i] = l1_norm(lij_ridge[i]-lij_mat)/l1_norm(lij_mat)
    mle_l1_abs[i] = l1_norm(lij_mle[i]-lij_mat)
    ridge_l1_abs[i] = l1_norm(lij_ridge[i]-lij_mat)

lasso_l1_rel = np.zeros(len(lasso_t_vals))
lasso_l1_abs = np.zeros(len(lasso_t_vals))
for i in range(len(lasso_t_vals)):
    lasso_l1_rel[i] = l1_norm(lij_lasso[i]-lij_mat)/l1_norm(lij_mat)
    lasso_l1_abs[i] = l1_norm(lij_lasso[i]-lij_mat)


# plot
plt.plot(t_vals[48:(nt-2)],mle_l1_rel[48:],label="MLE"); 
plt.plot(t_vals[48:(nt-2)],ridge_l1_rel[48:],label="RIDGE"); 
plt.plot(np.round(lasso_t_vals),lasso_l1_rel,label="LASSO")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_6e.pdf'
save_plot(fig_directory,filename)
plt.show()

plt.plot(t_vals[48:(nt-2)],mle_l1_abs[48:],label="MLE"); 
plt.plot(t_vals[48:(nt-2)],ridge_l1_abs[48:],label="RIDGE"); 
plt.plot(np.round(lasso_t_vals),lasso_l1_abs,label="LASSO")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'offline_est_sim_6f.pdf'
save_plot(fig_directory,filename)
plt.show()


#################################
           

################################
#### Lij Estimator (Online) ####
################################


## estimate simple mean field mat (constant & decreasing step sizes) ##
beta = 0.5; N = 5
aij_mat = mean_field_mat(N,beta)
lij_mat = laplacian_mat(N,aij_mat)

beta_init = 0.01
aij_mat_init = mean_field_mat(N,beta_init)
lij_mat_init = laplacian_mat(N,aij_mat_init)

alpha = 0.1; v_func = linear_func; sigma = 1; x0 =2; seed =1
T = 5000; dt = 0.1
nt = int(np.round(T/dt))
t_vals = np.linspace(0,T,nt+1)

step_size_init = 0.1; power = 0.75
step_size = step_size_decrease(T,dt,step_size_init,power)

n_seeds = 20
lij_online_fast = np.zeros((n_seeds,nt+1,N,N))
lij_online_med = np.zeros((n_seeds,nt+1,N,N))
lij_online_slow = np.zeros((n_seeds,nt+1,N,N))

## run sgd
for i in range(n_seeds):
    lij_online_fast[i,:,:,:] = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                                   Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                                   seed=i,step_size=step_size,method = "fast")
    
    lij_online_med[i,:,:,:] = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                                   Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                                   seed=i,step_size=step_size,method = "med")
    
    lij_online_slow[i,:,:,:] = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                                   Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                                   seed=i,step_size=step_size,method = "slow")
    

 
lij_online_fast = np.mean(lij_online_fast,axis=0)
lij_online_med = np.mean(lij_online_med,axis=0)
lij_online_slow = np.mean(lij_online_slow,axis=0)

## save!
#os.chdir(results_directory)
#np.save("lij_online_mean_field_fast",lij_online_fast)
#np.save("lij_online_mean_field_med",lij_online_med)
#np.save("lij_online_mean_field_slow",lij_online_slow)
#os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#lij_online_fast = np.load("lij_online_mean_field_fast.npy")
#lij_online_med = np.load("lij_online_mean_field_med.npy")
#lij_online_slow = np.load("lij_online_mean_field_slow.npy")
#os.chdir(default_directory)


  
## plots (fast, med, slow)   
for i in range(N):
    for j in range(i+1,N):
        plt.plot(t_vals[10:],lij_online_fast[10:,i,j])
        plt.axhline(y=lij_mat[i,j],color="black",linestyle="--")

plt.xlabel('t'); plt.ylabel(r'$\hat{\theta}(t)$')
filename = 'online_est_sim_3a.pdf'
save_plot(fig_directory,filename)
plt.show()

for i in range(N):
    for j in range(i+1,N):
        plt.plot(t_vals[10:],lij_online_med[10:,i,j])
        plt.axhline(y=lij_mat[i,j],color="black",linestyle="--")

plt.xlabel('t'); plt.ylabel(r'$\hat{\theta}(t)$')
filename = 'online_est_sim_3b.pdf'
save_plot(fig_directory,filename)
plt.show()

for i in range(N):
    for j in range(i+1,N):
        plt.plot(t_vals[10:],lij_online_slow[10:,i,j])
        plt.axhline(y=lij_mat[i,j],color="black",linestyle="--")

plt.xlabel('t'); plt.ylabel(r'$\hat{\theta}(t)$')
filename = 'online_est_sim_3c.pdf'
save_plot(fig_directory,filename)
plt.show()



# reconstruct estimate of interaction matrix
aij_online_fast = np.zeros((nt+1,N,N))
for i in range(nt+1):
    col_sums = mean_field_col_sums(N,lij_online_fast[i,:,:])
    aij_online_fast[i,:,:] = interaction_mat(N,lij_online_fast[i,:,:],col_sums)
  
aij_online_med = np.zeros((nt+1,N,N))
for i in range(nt+1):
    col_sums = mean_field_col_sums(N,lij_online_med[i,:,:])
    aij_online_med[i,:,:] = interaction_mat(N,lij_online_med[i,:,:],col_sums)
    
aij_online_slow = np.zeros((nt+1,N,N))
for i in range(nt+1):
    col_sums = mean_field_col_sums(N,lij_online_slow[i,:,:])
    aij_online_slow[i,:,:] = interaction_mat(N,lij_online_slow[i,:,:],col_sums)


beta_online_fast = N*np.mean(aij_online_fast,axis=(1,2))
beta_online_med = N*np.mean(aij_online_med,axis=(1,2))
beta_online_slow = N*np.mean(aij_online_slow,axis=(1,2))

# direct estimate
beta_online_direct = np.zeros((n_seeds,nt+1))
for i in range(n_seeds):
    beta_online_direct[i,:] = mean_field_rmle(N,T,v_func,alpha,beta,beta_init,
                                              sigma,x0,dt,seed,step_size=step_size)

beta_online_direct = np.mean(beta_online_direct,axis=0)

# plot estimates
plt.plot(t_vals,beta_online_fast,label="Indirect (All Constraints)")
plt.plot(t_vals,beta_online_med,label="Indirect (Symmetry Constaints)")
plt.plot(t_vals,beta_online_slow,label="Indirect (No Constraints)")
plt.plot(t_vals,beta_online_direct,label="Direct")
plt.axhline(y=beta,color="k",linestyle="--")
plt.legend()
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$')
filename = 'online_est_sim_3d.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot mse (small t)
t_init = 1; t_fin = 1000
plt.plot(t_vals[t_init:t_fin],(beta_online_fast[t_init:t_fin]-beta)**2,label="Indirect (All Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_med[t_init:t_fin]-beta)**2,label="Indirect (Symmetry Constaints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_slow[t_init:t_fin]-beta)**2,label="Indirect (No Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_direct[t_init:t_fin]-beta)**2,label="Direct")
plt.legend()
plt.xlabel('t'); plt.ylabel(r'L2 Error')
filename = 'online_est_sim_3e.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot mse (medium t)
t_init = 1000; t_fin = 20000
plt.plot(t_vals[t_init:t_fin],(beta_online_fast[t_init:t_fin]-beta)**2,label="Indirect (All Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_med[t_init:t_fin]-beta)**2,label="Indirect (Symmetry Constaints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_slow[t_init:t_fin]-beta)**2,label="Indirect (No Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_direct[t_init:t_fin]-beta)**2,label="Direct")
plt.legend()
plt.xlabel('t'); plt.ylabel(r'L2 Error')
filename = 'online_est_sim_3f.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot mse (large t)
t_init = 20000; t_fin = nt
plt.plot(t_vals[t_init:t_fin],(beta_online_fast[t_init:t_fin]-beta)**2,label="Indirect (All Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_med[t_init:t_fin]-beta)**2,label="Indirect (Symmetry Constaints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_slow[t_init:t_fin]-beta)**2,label="Indirect (No Constraints)")
plt.plot(t_vals[t_init:t_fin],(beta_online_direct[t_init:t_fin]-beta)**2,label="Direct")
plt.legend()
plt.xlabel('t'); plt.ylabel(r'L2 Error')
filename = 'online_est_sim_3g.pdf'
save_plot(fig_directory,filename)
plt.show()









## estimate sparse aij matrix using rmle ##
N = 10; T=2000; 
v_func=linear_func; alpha=0.1;
sigma=1; x0=2;dt=0.1; seed = 2

aij_mat = np.diag((1,1,1,1,1,2,2,2,2,2)) + np.diag((0.5,0.5,0.5,0.5,0.3,0.3,0.4,0.4,0.4),1) + np.diag((0.5,0.5,0.5,0.5,0.3,0.3,0.4,0.4,0.4),-1) +np.diag((0.4,0.4,0.5,0.5),6) + np.diag((0.4,0.4,0.5,0.5),-6)
lij_mat = laplacian_mat(N,aij_mat)

beta_init = 0.1
aij_mat_init = np.ones((N,N)).astype(float)
lij_mat_init = laplacian_mat(N,aij_mat_init)


nt = int(np.round(T/dt))
t_vals = np.linspace(0,T,nt+1)

step_size_init_fast = 0.5; power_fast = 0.85
step_size_fast = step_size_decrease(T,dt,step_size_init_fast,power_fast)

step_size_init_med = 0.35; power_med = 0.50
step_size_med = step_size_decrease(T,dt,step_size_init_med,power_med)

step_size_init_slow = 0.35; power_slow = 0.50
step_size_slow = step_size_decrease(T,dt,step_size_init_slow,power_slow)

## online estimator
lij_online_fast = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                               Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                               seed=seed,step_size=step_size_fast,
                               method = "fast")

lij_online_med = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                               Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                               seed=seed,step_size=step_size_med,
                               method = "med")

lij_online_slow = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                               Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                               seed=seed,step_size=step_size_slow,
                               method = "slow")

## offline  estimator
lij_offline = Lij_mle_ridge_lasso(N,T,v_func,alpha,beta,Lij=lij_mat,
                                  sigma=sigma,x0=x0,dt=dt,seed=seed,
                                  ridge_param=0,mle_ridge=True,lasso=False,
                                  lasso_param=0,Lij_init=None,
                                  lasso_t_vals=None)


## plots ##

# plot true matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mat,interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_4a.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot online estimate (fast)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_online[nt],interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_4b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot offline estimate
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_offline[nt-2],interpolation="nearest")
axes.set_xticks(np.arange(0,N,1))
axes.set_yticks(np.arange(0,N,1))
axes.set_xticklabels(np.linspace(1,N,N).astype(int))
axes.set_yticklabels(np.linspace(1,N,N).astype(int))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_4c.pdf'
save_plot(fig_directory,filename)
plt.show()


# l1 and l2 error
lij_online_fast_l2 = np.mean(abs(lij_online_fast-lij_mat)**2,axis=(1,2))
lij_online_fast_l1= np.mean(abs(lij_online_fast-lij_mat),axis=(1,2))

lij_online_med_l2 = np.mean(abs(lij_online_med-lij_mat)**2,axis=(1,2))
lij_online_med_l1= np.mean(abs(lij_online_med-lij_mat),axis=(1,2))

lij_online_slow_l2 = np.mean(abs(lij_online_slow-lij_mat)**2,axis=(1,2))
lij_online_slow_l1= np.mean(abs(lij_online_slow-lij_mat),axis=(1,2))

lij_offline_l2 = np.mean(abs(lij_offline-lij_mat)**2,axis=(1,2))
lij_offline_l1 = np.mean(abs(lij_offline-lij_mat),axis=(1,2))

# plot (l1 and) l2 error (all times)
t_init = 100; t_fin = nt-2
plt.plot(t_vals[t_init:t_fin],lij_online_fast_l1[t_init:t_fin],label=r"Online (All Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_online_med_l1[t_init:t_fin],label=r"Online (Symmetry Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_online_slow_l1[t_init:t_fin],label=r"Online (No Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_offline_l1[t_init:t_fin],label=r"Offline")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'online_est_sim_4d.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot (l1 and) l2 error (late times)
t_init = 10000; t_fin = nt-2
plt.plot(t_vals[t_init:t_fin],lij_online_fast_l1[t_init:t_fin],label=r"Online (All Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_online_med_l1[t_init:t_fin],label=r"Online (Symmetry Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_online_slow_l1[t_init:t_fin],label=r"Online (No Constraints)")
plt.plot(t_vals[t_init:t_fin],lij_offline_l1[t_init:t_fin],label=r"Offline")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'online_est_sim_4e.pdf'
save_plot(fig_directory,filename)
plt.show()






## estimate dense interaction matrix (not mean-field!)
N = 40

seed = 2
aij_mat = abs(np.random.randn(N,N))
aij_mat = 1/(N/10)*aij_mat
aij_mat = 0.5*(aij_mat+aij_mat.T)
np.fill_diagonal(aij_mat,0)

aij_mat_init = 1.5*deepcopy(aij_mat)

lij_mat = laplacian_mat(N,aij_mat)
lij_mat_init = laplacian_mat(N,aij_mat_init) 

alpha = 0.1; v_func = linear_func; sigma = 1; x0 = 2; seed = 3
T = 2500; dt = 0.1
nt = int(np.round(T/dt))
t_vals = np.linspace(0,T,nt+1)

step_size_init = .01; power = 0.50001; delay = 20
step_size = step_size_decrease(T,dt,step_size_init,power,delay)

method = "fast"

# rml
lij_online = Lij_rmle(N=N,T=T,v_func=v_func,alpha=alpha,Lij=lij_mat,
                      Lij_init=lij_mat_init,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,step_size=step_size,
                      method = method)

# mle
lij_offline = Lij_mle_ridge_lasso(N,T,v_func,alpha,beta,Lij=lij_mat,
                                  sigma=sigma,x0=x0,dt=dt,seed=seed,
                                  ridge_param=0,mle_ridge=True,lasso=False,
                                  lasso_param=0,Lij_init=None,
                                  lasso_t_vals=None)

# save!
#os.chdir(results_directory)
#np.save("lij_online_dense_large_N",lij_online)
#os.chdir(default_directory)

# reopen!
#os.chdir(results_directory)
#lij_online_test = np.load("lij_online_dense_large_N.npy")
#os.chdir(default_directory)
    
# 'final' estimates
lij_online_fin = np.mean(lij_online[nt-100:nt],axis=0)
lij_offline_fin = lij_offline[nt-2]

# reconstruct aij matrix
aij_online = np.zeros((nt+1,N,N))
for i in range(nt+1):
    col_sums = zero_diag_col_sums(N,lij_online[i,:,:])
    aij_online[i,:,:] = interaction_mat(N,lij_online[i,:,:],col_sums)
    np.fill_diagonal(aij_online[i,:,:],np.diag(aij_mat))

aij_offline = np.zeros((nt-1,N,N))
for i in range(nt-1):
    col_sums = zero_diag_col_sums(N,lij_offline[i,:,:])
    aij_offline[i,:,:] = interaction_mat(N,lij_offline[i,:,:],col_sums)
    
# 'final' estimates
aij_online_fin = np.mean(aij_online[(nt-100):nt],axis=0)
aij_offline_fin = aij_offline[nt-2]


# plot true lij matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_mat)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5a.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot online lij estimate (fast)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_online_fin)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot offline lij estimate
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(lij_offline_fin)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5c.pdf'
save_plot(fig_directory,filename)
plt.show()



# plot true aij matrix
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(aij_mat)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5d.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot online aij estimate (fast)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(aij_online_fin)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5e.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot offline aij estimate
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.matshow(aij_offline_fin)
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
filename = 'online_est_sim_5f.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot online aij l1 error
fig, axes = plt.subplots(nrows=1, ncols=1)
im = axes.matshow(abs(aij_online_fin-aij_mat),interpolation="nearest")
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
fig.colorbar(im)
filename = 'online_est_sim_5g.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot offline aij l1 error
fig, axes = plt.subplots(nrows=1, ncols=1)
im = axes.matshow(abs(aij_offline_fin-aij_mat),interpolation="nearest")
axes.set_xticks(np.arange(5,N+5,5)-1)
axes.set_yticks(np.arange(5,N+5,5)-1)
axes.set_xticklabels(np.arange(5,N+5,5))
axes.set_yticklabels(np.arange(5,N+5,5))
plt.gca().xaxis.tick_bottom()
fig.colorbar(im)
filename = 'online_est_sim_5h.pdf'
save_plot(fig_directory,filename)
plt.show()


# aij l1 error
aij_online_l1_abs = np.sum(abs(aij_online-aij_mat),axis=(1,2))
aij_online_l1_rel = aij_online_l1_abs/ np.sum(abs(aij_mat))
aij_offline_l1_abs = np.mean(abs(aij_offline-aij_mat),axis=(1,2))

# plot aij l1 error (online)
t_init = 100; t_fin = nt-2
plt.plot(t_vals[t_init:t_fin],1/(N**2)*aij_online_l1_abs[t_init:t_fin],label="Online",color="C0")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'online_est_sim_5i.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot aij l1 error (offline)
t_init = 100; t_fin = nt-2
plt.plot(t_vals[t_init:t_fin],1/(N**2)*aij_offline_l1_abs[t_init:t_fin],label="Offline",color="C1")
plt.xlabel('t'); plt.ylabel(r'L1 Error')
plt.legend()
filename = 'online_est_sim_5j.pdf'
save_plot(fig_directory,filename)
plt.show()


# compare simulated output
xt = sde_sim_func(N,T,v_func,alpha,beta=None,Aij=aij_mat,Lij=None,sigma=sigma,
                  x0=x0,dt=dt,seed=seed)

xt_est = sde_sim_func(N,T,v_func,alpha,beta=None,Aij=aij_online_fin,Lij=None,
                      sigma=sigma,x0=x0,dt=dt,seed=seed)


# plot histograms (or densities) for different particles
for i in range(9):
    sns.distplot(xt[:,i],hist=False,kde=True,label=r"$f(x|L_{ij})$")
    sns.distplot(xt_est[:,i],hist=False,label=r"$f(x|\hat{L}_{ij})$")
    plt.xlabel('$x$'); plt.ylabel(r'$f(x)$')
    plt.legend()
    filename = 'online_est_sim_5k_{}.pdf'.format(i+1)
    save_plot(fig_directory,filename)
    plt.show()

# ks test 
ks_test_p_vals = np.zeros(N)
for i in range(N):
    ks_test_p_vals[i] = round(sp.stats.kstest(xt[:,i],xt_est[:,i])[1],10)


################################





###############################################
#### Interaction Kernel Estimator (Online) ####
###############################################


## SINGLE BUMP KERNEL ##

## general parameters
T = 200; dt = 0.1; t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.2; 
seed = 2; v_func = null_func; 

## interaction parameters
Aij_calc = True; Aij_influence_func = influence_func_bump; 
Aij_calc_func = Aij_calc_func; 
Aij_scale_param = 2 #use 0.8 when using row sum normalisation in Aij

## step size parameters
step_size_init = 0.002
step_size = step_size_init
#step_size = step_size_decrease(T=T,dt=dt,step_size_init=step_size_init,power=.51)

## param estimation parameters
param_est_indices = [0]; n_param = 3; 
param_centre_true = [-0.5]#[-1,-0.75,-0.5];

## N values
N_vals = [2,3,4,5,8,10,15,20,25,30,40,50]#,10,25,50]

## number of random simulations
n_sims = 50

## width and squeeze
width_true = 2; squeeze_true = 0.01
    

## parameter estimation
all_param_est = np.zeros((int((np.round(T/dt))+1),n_sims,len(N_vals),len(param_centre_true)))

for i in range(len(N_vals)):
    
    ## value of N
    N = N_vals[i]
    
    ## initial conditions
    x0 = np.linspace(-2,2,N)
    
    for l in range(len(param_centre_true)):
    
        centre_true = param_centre_true[l]
        param_true = [centre_true,width_true,squeeze_true]; 
        centre = param_true[0]; width = param_true[1]; squeeze = param_true[2]; 
        
        ## initial parameters
        param_est_init = np.zeros((n_sims,3))
        param_est_init_centre = np.random.uniform(centre_true+1,centre_true+2,n_sims)
        for m in range(n_sims):
            param_est_init[m,0] = param_est_init_centre[m]
            param_est_init[m,1:3] = [2,.01]
    
        ## plot interaction kernel
        #x_vals = np.linspace(0,1.5,201)
        #y_vals = [influence_func_bump(x,centre,width,squeeze) for x in x_vals]
        #plt.plot(x_vals,y_vals)
        #plt.show()

        ## parameter estimation
        for k in range(n_sims):
            
            seed = k
            
            param_est,xt = Lij_rmle(N=N,T=T,v_func=null_func,alpha = alpha, Lij = None, 
                     Lij_init = None, sigma = sigma,x0 = x0,dt = dt, seed = seed,
                     step_size = step_size, method = "compute", aij_func = Aij_calc_func, 
                     kernel_func = influence_func_bump,
                     aij_grad_func = Aij_grad_calc_func, 
                     kernel_func_grad = influence_func_bump_grad, 
                     n_param = n_param, param_est_init = param_est_init[k,:], 
                     param_est_indices = param_est_indices,
                     aij_scale_param = Aij_scale_param,
                     param_true = param_true, centre = centre, width = width,
                     squeeze = squeeze)
            
            all_param_est[:,k,i,l] = param_est[:,0]
            
            if True:
                for j in range(0,N):
                        plt.plot(t_vals,xt[:,j],linewidth=1)
                        plt.xlabel(r'$t$'); plt.ylabel(r'$(t)$')
                plt.show()
                
                for j in range(0,n_param):
                    if j in param_est_indices:
                        plt.plot(t_vals,param_est[:,j],linewidth=1)
                        plt.axhline(param_true[j],color="C1",linestyle="--")
                        plt.xlabel(r'$t$'); plt.ylabel(r'${\theta(t)}$')
                plt.show()
        
            print(N,centre_true,k)
            
        ## plot individual estimates
        
        # weird estimates
        #l=9, k=22
        #l=8, k=16
            
        sims_to_plot = np.delete(all_param_est[:,:,i,l],[16,22],axis=1)
        n_sims_to_plot = sims_to_plot.shape[1]
        
        for k in range(n_sims_to_plot):
            plt.plot(t_vals,sims_to_plot[:,k]+0.5*width,color='C0',linewidth=0.8)
            plt.axhline(param_centre_true[l]+0.5*width,color="C1",linestyle="--",linewidth=1.5)
            plt.xlabel(r'$t$'); plt.ylabel(r'$\theta_t$')
            plt.ylim([0.1,2.6])
    
        ## plot mean estimate
        plt.plot(t_vals,np.mean(sims_to_plot[:,:],axis=1)+0.5*width,color="black",linewidth=3)
        plt.plot(t_vals,np.mean(sims_to_plot[:,:],axis=1)+0.5*width+np.sqrt(np.var(all_param_est[:,:,i,l],axis=1)),color="black",linewidth=2,linestyle="--")
        plt.plot(t_vals,np.mean(sims_to_plot[:,:],axis=1)+0.5*width-np.sqrt(np.var(all_param_est[:,:,i,l],axis=1)),color="black",linewidth=2,linestyle="--")
        plt.legend([r"$N={}$".format(N_vals[i])])
        
        ## save
        N = N_vals[i]
        param_centre = param_centre_true[l]
        filename = 'online_est_sim_7_T_500_N_{}_centre_{}_n_samples_{}_v2.pdf'.format(N,param_centre,n_sims)
        #save_plot(fig_directory,filename)
        plt.show()
            
## save results
os.chdir(results_directory)
filename = "T200_N_2_3_4_5_8_10_15_20_25_30_40_dt0_1_Aij_scale2_normN_step_size0_001_centre0_and_others_width1_squeeze0_01_x0_-2to2_sigma0_2_n_sims_50"
#np.save(filename,all_param_est)
os.chdir(default_directory)
    
        
## additional plots for paper (some sample paths, different values of $N$)
x_vals = np.linspace(0,1.5,151)
y_vals = [influence_func_bump(x,0,width_true,squeeze_true) for x in x_vals]
plt.plot(x_vals,y_vals)
plt.show()
    
for i in range(len(N_vals)):
    N=N_vals[i]
    x0 = np.linspace(-2,2,N)
    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      Aij_scale_param = Aij_scale_param,
                      centre = centre, width=width, squeeze=squeeze)
    plt.plot(t_vals,xt)
    plt.xlabel(r"$x_t$")
    plt.ylabel(r"t")
    filename = 'online_est_sim_7_T_500_N_{}_centre_{}_sample_paths.pdf'.format(N,param_centre)
    save_plot(fig_directory,filename)
    plt.show()
    print(N)
    

## additional plots for presentation: sample paths for different
## values of centre
param_centre_true = [-1.0,-0.8,-0.7,-0.5,-0.3,0.5]#,-0.4,-0.2,0,0.2,0.4]
T = 100; dt = 0.1; t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
sigma = 0.09


N=25
for z in param_centre_true:
    centre = z
    #x_vals = np.linspace(0,1.5,151)
    #y_vals = [influence_func_bump(x,z,width_true,squeeze_true) for x in x_vals]
    #plt.plot(x_vals,y_vals)
    #plt.show()
    #print(centre)
    width = width_true
    squeeze = squeeze_true
    np.random.seed(0)
    x0 = np.linspace(-2,2,N)
    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      Aij_scale_param = Aij_scale_param,
                      centre = centre, width=width, squeeze=squeeze)
    plt.plot(t_vals,xt)
    plt.xlabel(r"$x_t$")
    plt.ylabel(r"t")
    plt.ylim([-3.2,3.8])
    filename = 'online_est_sim_7_T_{}_N_{}_centre_{}_width{}_squeeze_{}_sample_paths.pdf'.format(T,N,z,width_true,squeeze_true)
    save_plot(fig_directory,filename)
    plt.show()
    
    
## additional plots for presentation: sample paths for different
## values of N
centre_true = -0.5
N_vals = [2,5,10,15,25,50]
T = 200; dt = 0.1; t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
sigma = 0.09


for z in N_vals:
    N=z
    #x_vals = np.linspace(0,1.5,151)
    #y_vals = [influence_func_bump(x,z,width_true,squeeze_true) for x in x_vals]
    #plt.plot(x_vals,y_vals)
    #plt.show()
    #print(centre)
    centre = centre_true
    width = width_true
    squeeze = squeeze_true
    np.random.seed(0)
    x0 = np.linspace(-2,2,N)
    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      Aij_scale_param = Aij_scale_param,
                      centre = centre, width=width, squeeze=squeeze)
    plt.plot(t_vals,xt)
    plt.xlabel(r"$x_t$")
    plt.ylabel(r"t")
    plt.ylim([-3.5,3.5])
    filename = 'online_est_sim_7_T_{}_N_{}_centre_{}_width{}_squeeze_{}_sigma_{},sample_paths.pdf'.format(T,z,centre_true,width_true,squeeze_true,sigma)
    save_plot(fig_directory,filename)
    plt.show()
    
    




#### we're finally getting some decent results! 

#### some important points to note

#### the likelihood function has relatively flat maxima, meaning the online
#### method in particular may struggle to locate the true maxima / be slow to
#### converge

#### the gradient of the likelihood function varies very very significantly 
#### on a local scale (see param_est_interac_testing.py), introudcing many 
#### many local minima and maxima that the SGD algorithm could get trapped in

#### there is an issue relating to identifiability, since once cluster 
#### formation or consensus has taken place, many interaction functions, i.e., 
#### interaction functions with different values of the 'centre' parameter
#### are almost as likely as one other, i.e., they generate essentially the 
#### same dynamics, since from this point foward we only observe very
#### short range interactions


#### some older observations & speculations

#### as noted above, the online method often doesn't work very for well
#### estimating a 'flat' interaction function with compact support

#### indeed, at any given time instant, there is no strong evidence in favour
#### of one value of the 'centre' parameter vs another, which is another 
#### way of saying that this parameter isn't really identifiable in the online
#### setting

#### indeed, once clusters have been formed, any parameter which permits the 
#### formation of clusters separated by those distances is almost as  likely
#### and so the log-likelihood will be almost completely flat, and its gradient
#### very close to zero; we see this in some of our simulations

#### speaking broadly, we could argue that it is the evolution of the system 
#### over a particular time period (i.e., the entire trajectories) which
#### allows the attraction parameter to be identified; this is why the 
#### (offline) likelihood does, in general, have a maximum at the true 
#### parameter value (see param_est_interac_testing.py) but often online 
#### parameter estimation fails

#### another consequence of these observations is that, as T increases, the 
#### performance of an offline maximum likelihood estimator could decrease, 
#### though it's not actually clear whether this is the case; the reasoning 
#### for this goes as follows:mfor larger values of T, the likelihood will be 
#### more heavily weighted towards the asymptotic trajectories of the 
#### particles, which, as we noted above, are almost as likely under many 
#### possible different values of the  parameter (i.e. any values of the 
#### parameter which are admissible for the with respect to the asymptotic 
#### cluster formation); at the very least, we would perhaps expect for 
#### maxima of the likelihood to become flatter as T increases

#### on the above point, it would be interesting to plot the log-likelihood 
#### as a function of T, and observe how it changes as time processes (we 
#### would expect it to `flatten out' as T increases, but perhaps to get 
#### flat as N increases; this is quite interesting, as it would suggest
#### a trade-off between the N and T asymptotics; large N favours good
#### estimation, while large T does not)

####  we could try to run SGD multiple times over the intiial trajectories of
#### many particles, as these parts of the trajectories seem to be the most
#### informative with respect to identifying this parameter


## TWO BUMP KERNELS ##


## general parameters
N = 40; T = 100; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.3; 
x0 = np.linspace(0,6,N); dt = 0.05; seed = 2; v_func = null_func; 

## interaction parameters
Aij_calc = True; Aij_influence_func = influence_func_mult_bump; 
Aij_calc_func = Aij_calc_func; 
Aij_scale_param = 2 #use 0.8 when using row sum normalisation in Aij

## param estimation parameters
param_est_indices = [0,3]; param_est_init = [1,1,.01,-0.4,1,0.01]
n_param = 6; param_true = [-0.1,1,0.01,0.4,1,0.01]; 
centre = [param_true[0],param_true[3]]; 
width = [param_true[1],param_true[4]]; 
squeeze = [param_true[2],param_true[5]]; 

## plot interaction kernel
x_vals = np.linspace(0,1.5,201)
y_vals = [influence_func_mult_bump(x,centre,width,squeeze) for x in x_vals]
plt.plot(x_vals,y_vals)
plt.show()

## step size parameters
step_size_init = 0.002
step_size = step_size_init
#step_size = step_size_decrease(T=T,dt=dt,step_size_init=step_size_init,power=.5001)

## parameter estimation
n_sims = 20
all_param_est = np.zeros((int((np.round(T/dt))+1),len(param_est_indices),n_sims))

for k in range(n_sims):
    
    seed = k;
    
    param_est,xt = Lij_rmle(N=N,T=T,v_func=null_func,alpha = alpha, Lij = None, 
             Lij_init = None, sigma = sigma,x0 = x0,dt = dt, seed = seed,
             step_size = step_size, method = "compute", aij_func = Aij_calc_func, 
             kernel_func = influence_func_mult_bump,
             aij_grad_func = Aij_grad_calc_func, 
             kernel_func_grad = influence_func_mult_bump_grad, 
             n_param = n_param, param_est_init = param_est_init, 
             param_est_indices = param_est_indices,
             aij_scale_param = Aij_scale_param,
             param_true = param_true, centre = centre, width = width,
             squeeze = squeeze)
    
    all_param_est[:,:,k] = param_est[:,param_est_indices]
    
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    for j in range(0,N):
            plt.plot(t_vals,xt[:,j],linewidth=1)
            plt.xlabel(r'$t$'); plt.ylabel(r'$(t)$')
    plt.show()
    
    for j in range(0,n_param):
        if j in param_est_indices:
            plt.plot(t_vals,param_est[:,j],linewidth=1)
            plt.axhline(param_true[j],color="C2",linestyle="--")
            plt.xlabel(r'$t$'); plt.ylabel(r'${\theta(t)}$')
    plt.show()

for k in range(n_sims):
    if k not in [12,15]:
        plt.plot(t_vals,all_param_est[:,0,k]+0.5*width[0],color='C0',linewidth=0.9)
        plt.plot(t_vals,all_param_est[:,1,k]+0.5*width[0],color='C1',linewidth=0.9)
        plt.axhline(param_true[0]+0.5*width[0],color="C2",linestyle="--")
        plt.axhline(param_true[3]+0.5*width[0],color="C2",linestyle="--")
        plt.xlabel(r'$t$'); plt.ylabel(r'${\theta(t)}$')
plt.plot(t_vals,np.mean(all_param_est,axis=2)+0.5*width[0],color="black",linewidth=3)
filename = 'online_est_sim_8_2.pdf'.format(i)
save_plot(fig_directory,filename)
plt.show()


## save simulations!
if False:
    
    os.chdir(results_directory)
    np.save("all_param_est_mult_bump_2",all_param_est)
    os.chdir(default_directory)
    
    ## these were the settings!
    
    
    ## o.g. simulation
    
    ## general parameters
    #N = 40; T = 100; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.2; 
    #x0 = np.linspace(0,5,N); dt = 0.1; seed = 2; v_func = null_func; 
    
    ## interaction parameters
    #Aij_calc = True; Aij_influence_func = influence_func_mult_bump; 
    #Aij_calc_func = Aij_calc_func; 
    #Aij_scale_param = 2 #use 0.8 when using row sum normalisation in Aij
    
    ## param estimation parameters
    #param_est_indices = [0,3]; param_est_init = [1,1,.01,-0.4,1,0.01]
    #n_param = 6; param_true = [-0.1,1,0.01,0.4,1,0.01]; 
    #centre = [param_true[0],param_true[3]]; 
    #width = [param_true[1],param_true[4]]; 
    #squeeze = [param_true[2],param_true[5]]; 

    
    ## step size parameters
    #step_size_init = 0.002
    #step_size = step_size_init
    #step_size = step_size_decrease(T=T,dt=dt,step_size_init=step_size_init,power=.5001)
    
    ## number of sims 
    # n_sims = 20
    
    
    
    ## simulation _1

    # change initial conditions to
    # x0 = np.linspace(0,10,N)
    
    
    ## simulation _2
    
    # change initial conditions to 
    # x0 = np.linspace(0,6,N)
    # sigma = 0.3
    # dt = 0.05
    

###############################################
