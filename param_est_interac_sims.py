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
import sys
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from param_est_interac_funcs import * 
import timeit as timeit
import seaborn as sns
import os
import array_to_latex as atl

default_directory = "/Users/ls616"
code_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/Project Ideas/code"
results_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/Project Ideas/code/results"
fig_directory = "/Users/ls616/Google Drive/MPE CDT/PhD/Year 3/Project Ideas/notes/figures/sde_sims"
sys.path.append(code_directory)

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
N = 5; T = 400; alpha = 0.1;
beta_vals = np.array([0.00,0.01,0.05,0.1,0.2,0.5,1.0]);                    
sigma = 1; x0 = 1; dt = 0.1; seed = 2
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)

## Simulation
for i in range(beta_vals.shape[0]):
    sim_test = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,
                            beta=beta_vals[i],Aij = None, Lij = None,
                            sigma=sigma,x0=x0,dt=dt,seed=seed)
    plt.plot(t_vals,sim_test)
    plt.plot(t_vals,np.mean(sim_test,axis=1),color='black',linewidth=2.0)
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

beta_est_mle = np.zeros((nt-1,n_seeds))
for i in range(n_seeds):
    beta_est_mle[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=1,sigma=1,x0=2,dt=0.1,seed=i,
                                  mle=True)[0]


beta_est_map = np.zeros((nt-1,n_seeds))
for i in range(n_seeds):
    beta_est_map[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=1,sigma=1,x0=2,dt=0.1,seed=i,
                                  mle=False)[0]

## plot MLE estimates
plt.plot(t_vals[100:],beta_est_mle[100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_mle[100:],axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MLE}}(t)$');
filename = 'offline_est_sim_1a.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates
plt.plot(t_vals[100:],beta_est_map[100:],linewidth=0.9); 
plt.plot(t_vals[100:],np.mean(beta_est_map[100:],axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_1b.pdf'
save_plot(fig_directory,filename)
plt.show()

# plot l1 errors
plt.plot(t_vals[100:],abs(beta-beta_est_mle[100:,]))
plt.plot(t_vals[100:],np.mean(abs(beta-beta_est_mle[100:,]),axis=1),color="black",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}_{\mathrm{MLE}}|$')
filename = 'offline_est_sim_1c.pdf'
save_plot(fig_directory,filename)
plt.show()


plt.plot(t_vals[100:],abs(beta-beta_est_map[100:,]))
plt.plot(t_vals[100:],np.mean(abs(beta-beta_est_mle[100:,]),axis=1),color="black",linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}_{\mathrm{MAP}}|$')
filename = 'offline_est_sim_1d.pdf'
save_plot(fig_directory,filename)
plt.show()


# plot mse (very similar, MAP slightly better for small T)
plt.plot(t_vals[100:],np.mean((beta-beta_est_mle[100:])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MLE}}$")#,color="black",linewidth=2.0)
plt.plot(t_vals[100:],np.mean((beta-beta_est_map[100:])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MAP}}$")#,color="grey",linewidth=2.0)
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1e.pdf'
save_plot(fig_directory,filename)
plt.show()

plt.plot(t_vals[0:20],np.mean((beta-beta_est_mle[0:20])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MLE}}$")
plt.plot(t_vals[0:20],np.mean((beta-beta_est_map[0:20])**2,axis=1),label=r"$\hat{\beta}_{\mathrm{MAP}}$")
plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta})^2\right]$')
plt.legend()
filename = 'offline_est_sim_1f.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot variance (pretty much identical)
plt.plot(t_vals[100:],np.var(beta_est_mle[100:],axis=1))
plt.plot(t_vals[100:],np.var(beta_est_map[100:],axis=1))
plt.show()




# Estimate mean field parameter for different values of beta (MLE and MAP)

# -> estimation successful for all values

N=10;T=200;v_func=linear_func;alpha=0.1;
beta_vals=np.linspace(0,2,5);sigma=1
x0=2;dt=0.1;seed=2;
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

beta_est_mle = np.zeros((nt-1,beta_vals.shape[0]))
for i in range(beta_vals.shape[0]):
    beta_est_mle[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=beta_vals[i],sigma=1,x0=2,dt=0.1,seed=1,
                                  mle=True)[0]


beta_est_map = np.zeros((nt-1,beta_vals.shape[0]))
for i in range(beta_vals.shape[0]):
    beta_est_map[:,i] = mean_field_mle_map(N=N,T=T,v_func=v_func,alpha=alpha,
                                  beta=beta_vals[i],sigma=1,x0=2,dt=0.1,seed=1,
                                  mle=False)[0]
   
## plot MLE estimates
plt.plot(t_vals[20:],beta_est_mle[20:]); 
for i in range(beta_vals.shape[0]): plt.axhline(y=beta_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MLE}}(t)$')
filename = 'offline_est_sim_2a.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MAP estimates
plt.plot(t_vals[20:],beta_est_map[20:]); 
for i in range(beta_vals.shape[0]): plt.axhline(y=beta_vals[i],color='k',linewidth=0.9,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}_{\mathrm{MAP}}(t)$')
filename = 'offline_est_sim_2b.pdf'
save_plot(fig_directory,filename)
plt.show()




# Estimate mean field parameter for different number of particles (MLE and MAP); 
# trial averaged over several random seeds.

# -> in general, MSE and variance decrease as number of particles increases
# -> the improvement is particularly noticeable for small T
# -> no significant difference between MLE and MAP, especially as T increasees
# -> MAP does perform better for small values of T
# -> NB: this code can take a while to run!

N_vals=np.linspace(10,50,5);T=100;v_func=linear_func;alpha=0.1;
beta=1;sigma=1; x0=2;dt=0.1;seed_vals=np.linspace(1,10,10);
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

## compute MLE
beta_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        beta_est_mle[:,i,j] = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                               v_func=v_func,alpha=alpha,
                                               beta=beta,sigma=sigma,x0=x0,
                                               dt=dt,seed=np.int(seed_vals[j]),
                                               mle=True)[0]

## compute MAP
beta_est_map = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        beta_est_map[:,i,j] = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                               v_func=v_func,alpha=alpha,
                                               beta=beta,sigma=sigma,x0=x0,
                                               dt=dt,seed=np.int(seed_vals[j]),
                                               mle=False)[0]
   
## plot MLE estimates
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        plt.plot(t_vals[10:],beta_est_mle[10:,i,j])
    plt.plot(t_vals[10:],np.mean(beta_est_mle[10:,i,:],axis=1),linewidth=2,color='k',label="N={}".format(N_vals[i]));
    plt.axhline(y=1,color='k',linewidth=0.9,linestyle="--")
    plt.xlabel('t'); plt.ylabel(r'$\hat{\theta}(t)$')
    plt.legend()
    #filename = 'offline_est_sim_3_{}.pdf'.format(i)
    #save_plot(fig_directory,filename)
    #plt.show()
    
## plot MAP estimates
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        plt.plot(t_vals[10:],beta_est_map[10:,i,j])
    plt.plot(t_vals[10:],np.mean(beta_est_map[10:,i,:],axis=1),linewidth=2,color='k');
    plt.axhline(y=1,color='k',linewidth=0.9,linestyle="--")
    plt.xlabel('t'); plt.ylabel(r'$\hat{\theta}(t)$')
    plt.show()
    
       
## plot MLE normalised MSE (larger t)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.mean((beta - beta_est_mle[100:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()  
filename = 'offline_est_sim_3a.pdf'
save_plot(fig_directory,filename)
plt.show()
    

## plot MAP normalised MSE (larger t)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.mean((beta - beta_est_map[100:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()
filename = 'offline_est_sim_3b.pdf'
save_plot(fig_directory,filename)
plt.show()


## plot MLE normalised MSE (smaller t)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((beta - beta_est_mle[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()  
filename = 'offline_est_sim_3c.pdf'
save_plot(fig_directory,filename)
plt.show()
    

## plot MAP normalised MSE (smaller t)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[5:49],np.mean((beta - beta_est_map[5:49,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\theta-\hat{\theta}(t))^2\right]$')
    plt.legend()
filename = 'offline_est_sim_3d.pdf'
save_plot(fig_directory,filename)
plt.show()


## plot MLE variance (very similar to MSE)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(beta_est_mle[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\theta}(t))$')
    plt.legend()
plt.show()
    

## plot MAP variance (very similar to MSE)
for i in range(N_vals.shape[0]):
    plt.plot(t_vals[100:],np.var(beta_est_map[100:,i,],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\theta}(t))$')
    plt.legend()
plt.show()




# Compare MSE of MLE as function of N (fixed T)

N_vals=np.linspace(2,100,50);T=10;v_func=linear_func;alpha=0.1;
beta=1;sigma=1; x0=2;dt=0.1;seed_vals=np.linspace(1,200,200);
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
t_vals = t_vals[1:t_vals.shape[0]-1]
nt = np.int(round(T/dt))

## compute MLE
beta_est_mle = np.zeros((nt-1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        beta_est_mle[:,i,j] = mean_field_mle_map(N=np.int(N_vals[i]),T=T,
                                               v_func=v_func,alpha=alpha,
                                               beta=beta,sigma=sigma,x0=x0,
                                               dt=dt,seed=np.int(seed_vals[j]),
                                               mle=True)[0]
    print(i)

   
## save!
#os.chdir(results_directory)
#np.save("mean_field_offline_l1_error_vs_N",beta_est_mle)
#os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#beta_est_mle = np.load("mean_field_offline_l1_error_vs_N.npy")
#os.chdir(default_directory)

## plot 
beta_est_l1 = np.mean(abs(beta-beta_est_mle[nt-2]),axis=1)
beta_est_l1_upper = beta_est_l1 + 1.96*np.var(beta_est_mle[nt-2],axis=1)
beta_est_l1_lower = beta_est_l1 - 1.96*np.var(beta_est_mle[nt-2],axis=1)
plt.plot(N_vals[2:],beta_est_l1[2:])
plt.plot(N_vals[2:],beta_est_l1_upper[2:],color="C1",linestyle="--")
plt.plot(N_vals[2:],beta_est_l1_lower[2:],color="C1",linestyle="--")
plt.xlabel('$N$'); plt.ylabel(r'L1 Error') 
filename = 'offline_est_sim_7a.pdf'
save_plot(fig_directory,filename)
plt.show()

########################################




#######################################
#### Mean Field Estimator (Online) ####
#######################################

# estimate mean field parameter & average over different random seeds
# w. decreasing step size

N = 10; T = 1000; v_func = linear_func; alpha = 0.1; beta = 1;
sigma = 1; x0 = 2; dt = 0.1; 

n_seeds = 20

np.random.seed(1)
beta0 = np.random.uniform(-1,3,n_seeds); # randomly sample beta0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = int(np.round(T/dt))

## step sizes
step_size = np.zeros(nt)
step_size_init = 0.1
for i in range(nt):
    step_size[i] = min(step_size_init,step_size_init/(t_vals[i]**0.9))

## RML estimate
beta_est_rml = np.zeros((nt+1,n_seeds))
for i in range(n_seeds):
    beta_est_rml[:,i] = mean_field_rmle(N=N,T=T,v_func=v_func,alpha=alpha,
                                        beta=beta,beta0=beta0[i],sigma=sigma,
                                        x0=x0,dt=dt,seed=i,
                                        step_size=step_size)[0]


## plot MLE estimates
plt.plot(t_vals[100:],beta_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(beta_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$');
filename = 'online_est_sim_1a.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot L1 error
plt.plot(t_vals[200:],abs(beta-beta_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(beta-beta_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\beta-\hat{\beta}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}(t)|$');
plt.legend()
filename = 'online_est_sim_1b.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MSE 
plt.plot(t_vals[200:],(beta-beta_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((beta-beta_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\beta-\hat{\beta}(t))^2$');
plt.legend()
filename = 'online_est_sim_1c.pdf'
save_plot(fig_directory,filename)
plt.show()


# estimate mean field parameter & average over different random seeds
# w. constannt step size
N = 10; T = 1000; v_func = linear_func; alpha = 0.1; beta = 1;
sigma = 1; x0 = 2; dt = 0.1; 

n_seeds = 20

np.random.seed(1)
beta0 = np.random.uniform(-1,3,n_seeds); # randomly sample beta0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = int(np.round(T/dt))

## step sizes
step_size = 0.01

## RML estimate
beta_est_rml = np.zeros((nt+1,n_seeds))
for i in range(n_seeds):
    beta_est_rml[:,i] = mean_field_rmle(N=N,T=T,v_func=v_func,alpha=alpha,
                                        beta=beta,beta0=beta0[i],sigma=sigma,
                                        x0=x0,dt=dt,seed=i,
                                        step_size=step_size)[0]


## plot MLE estimates
plt.plot(t_vals[100:],beta_est_rml[100:,],linewidth=0.9); 
plt.plot(t_vals,np.mean(beta_est_rml,axis=1),linewidth=2,color='k');
plt.axhline(y=beta,color='k',linewidth=2,linestyle="--")
plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$');
filename = 'online_est_sim_1d.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot L1 error
plt.plot(t_vals[200:],abs(beta-beta_est_rml[200:,]))
plt.plot(t_vals[200:],np.mean(abs(beta-beta_est_rml[200:,]),axis=1),label=r"$\mathbb{E}\left|\beta-\hat{\beta}(t)\right|$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$|\beta-\hat{\beta}(t)|$');
plt.legend()
filename = 'online_est_sim_1e.pdf'
save_plot(fig_directory,filename)
plt.show()

## plot MSE 
plt.plot(t_vals[200:],(beta-beta_est_rml[200:,])**2)
plt.plot(t_vals[200:],np.mean((beta-beta_est_rml[200:,])**2,axis=1),label=r"$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$",color='k',linewidth=2)
plt.xlabel('t'); plt.ylabel(r'$(\beta-\hat{\beta}(t))^2$');
plt.legend()
filename = 'online_est_sim_1f.pdf'
save_plot(fig_directory,filename)
plt.show()




# Estimate mean field parameter for different number of particles; and average
# over different random seeds.

N_vals=np.linspace(10,50,5);T=100; v_func=linear_func; alpha=0.1; beta = 1;
sigma=1; x0=2; dt=0.01; seed_vals=np.linspace(1,10,10);

np.random.seed(1)
beta0 = np.random.uniform(-0.5,0.5,seed_vals.shape[0]); # randomly sample beta0

t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = np.int(round(T/dt))

## step sizes
step_size = np.zeros(nt)
step_size_init = 0.2
for i in range(nt):
    step_size[i] = min(step_size_init,step_size_init/(t_vals[i]**0.9))
    

## RML estimate
beta_est_rml = np.zeros((nt+1,N_vals.shape[0],seed_vals.shape[0]))
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        beta_est_rml[:,i,j] = mean_field_rmle(N=np.int(N_vals[i]),T=T,
                                              v_func=v_func,alpha=alpha,
                                              beta=beta,beta0=beta0[j],
                                              sigma=sigma,x0=x0,dt=dt,
                                              seed=np.int(seed_vals[j]))[0]
   
## plot estimates
for i in range(N_vals.shape[0]):
    for j in range(seed_vals.shape[0]):
        plt.plot(t_vals,beta_est_rml[:,i,j])
    plt.plot(t_vals,np.mean(beta_est_rml[:,i,:],axis=1),linewidth=2,color='k');
    plt.axhline(y=1,color='k',linewidth=0.9,linestyle="--")
    plt.xlabel('t'); plt.ylabel(r'$\hat{\beta}(t)$')
    filename = 'online_est_sim_2a_{}.pdf'.format(i)
    save_plot(fig_directory,filename)
    plt.show()
    

       
## plot MSE
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.mean((beta - beta_est_rml[:,i,:])**2,axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}\left[(\beta-\hat{\beta}(t))^2\right]$')
    plt.legend()
filename = 'online_est_sim_2b.pdf'
save_plot(fig_directory,filename)
plt.show()


## plot variance
for i in range(N_vals.shape[0]):
    plt.plot(t_vals,np.var(beta_est_rml[:,i,:],axis=1),label="N=%d" %N_vals[i])
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Var}(\hat{\beta}(t))$')
    plt.legend()
filename = 'online_est_sim_2c.pdf'
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
                                              sigma,x0,dt,seed,step_size=step_size)[0]

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






