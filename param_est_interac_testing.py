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
from mpl_toolkits.mplot3d import Axes3D
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


#################################
#### Test gradient functions ####
#################################

## Gradient of Gaussian interaction function ##

interaction_func = influence_func_exp
interaction_func_grad = influence_func_exp_grad

## with respect to mu 
test_param_index = 0
test_vals = np.linspace(-5,5,1001)
r_val = 1; sd_val = 1
y_vals = [interaction_func(r=r_val,mu=x,sd=sd_val) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,mu=x,sd=sd_val)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check)

## with respect to sigma^2
test_param_index = 1
test_vals = np.linspace(0.01,2.00,1000)
r_val = 0.5; mu_val = 1
y_vals = [interaction_func(r=r_val,mu=mu_val,sd=np.sqrt(x)) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,mu=mu_val,sd=np.sqrt(x))[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check)
plt.show()


## Gradient of bump interaction function ##

interaction_func = influence_func_bump
interaction_func_grad = influence_func_bump_grad

## with respect to centre 
test_param_index = 0
test_vals = np.linspace(0.2,0.8,1001)
r_val = 0.01; width_val = 1; squeeze_val = 0.1
y_vals = [interaction_func(r=r_val,centre=x,width = width_val, squeeze = squeeze_val) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,centre=x,width = width_val, squeeze = squeeze_val)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check)

## with respect to width
test_param_index = 1
test_vals = np.linspace(.5,5,1001)
r_val = 0.5; centre_val = 1; squeeze_val = 1
y_vals = [interaction_func(r=r_val,centre=centre_val,width = x, squeeze = squeeze_val) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,centre=centre_val,width = x, squeeze = squeeze_val)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check)


## with respect to squeeze
test_param_index = 2
test_vals = np.linspace(.01,3,1001)
r_val = 0.8; centre_val = 1; width_val = 1
y_vals = [interaction_func(r=r_val,centre=centre_val,width = width_val, squeeze = x) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,centre=centre_val,width = width_val, squeeze = x)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check)




## Gradient of mutliple bump interaction function ##

interaction_func = influence_func_mult_bump
interaction_func_grad = influence_func_mult_bump_grad

n_kernels = 2
n_centres = 1
n_test_points = 1001


### just check w.r.t. 'centre' parameters ##

## with respect to 1st centre ## 

## index & range of 1st centre
test_param_index = 0
test_vals = np.linspace(-0.3,+0.3,1001)

## fix other parameters
r_val = 0.01; 
centre2_val = 0.5; width_val = [1]*n_kernels; squeeze_val = [.01]*n_kernels

y_vals = [interaction_func(r=r_val,centre=[x,centre2_val],width = width_val, squeeze = squeeze_val) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,centre=[x,centre2_val],width = width_val, squeeze = squeeze_val)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.show()
plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check); plt.show()



## with respect to 2nd centre ## 

## index & range of 2nd centre
test_param_index = 3
test_vals = np.linspace(0.2,0.8,1001)

## fix other parameters
r_val = 0.4; 
centre1_val = 0; width_val = [1]*n_kernels; squeeze_val = [.01]*n_kernels

y_vals = [interaction_func(r=r_val,centre=[centre1_val,x],width = width_val, squeeze = squeeze_val) for x in test_vals]
y_grad_vals = [interaction_func_grad(r=r_val,centre=[centre1_val,x],width = width_val, squeeze = squeeze_val)[test_param_index] for x in test_vals]
y_grad_vals_check = np.diff(y_vals)/np.diff(test_vals)
plt.plot(test_vals,y_vals); plt.show()
plt.plot(test_vals,y_grad_vals); plt.plot(test_vals[1:],y_grad_vals_check); plt.show()


#################################




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
#plt.plot(sim_test1)

## should all overlap
plt.plot(sim_test1[:,1])
plt.plot(sim_test2[:,1])
plt.plot(sim_test3[:,1])

## should be (approx) zero
np.sum(sim_test1 - sim_test2,axis=None)
np.sum(sim_test1 - sim_test3,axis=None)

###########################




##########################
#### Test log_lik_lij ####
##########################


### For simple mean field model ###

## beta
beta_true = [0.1]#[0.25,0.5,0.75,1,1.25,1.5,1.75]
alpha_true = 0
beta_vals = np.linspace(-0.2,0.4,61)

N_vals = [2,5,10,25,50,100]

T=100;
n_runs = 20
seed_vals=np.linspace(1,n_runs,n_runs);
sigma = 1; x0 = 2; dt = 0.1; 
v_func = linear_func;

lls = np.zeros((len(beta_true),len(beta_vals),len(N_vals),T+1,n_runs))
        
for l in range(n_runs):
    for m in range(len(N_vals)):
        
        for j in range(len(beta_true)):
            
            N = N_vals[m];  beta = beta_true[j]; 
            Aij = None; Lij = None; Aij_calc = False
            alpha = alpha_true; 
            xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,Aij=Aij,
                              Lij=Lij,sigma=sigma,x0=x0,dt=dt,seed=np.int(seed_vals[l]),Aij_calc=Aij_calc)
            
            #plt.plot(xt)
            #plt.show()
            
            dxt = np.diff(xt,axis=0)
            
            for i in range(len(beta_vals)):
                Lij = laplacian_mat(N,mean_field_mat(N,beta_vals[i]))
                lls[j,i,m,:,l] = log_lik_lij(Lij = Lij,N=N,t=T,xt=xt,dxt=dxt,v_func=v_func,
                                       alpha = alpha,sigma=sigma,dt=dt,
                                       Aij_calc=Aij_calc,marginal=True)[1]
    
    ## for sanity
    print(l)


## save!
#os.chdir(results_directory)
#filename = 'online_est_sim_log_lik_beta_plots.pdf'
#np.save(filename,lls)
#os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#filename = 'online_est_sim_log_lik_beta_plots.npy'
#lls = np.load(filename)
#os.chdir(default_directory) 


## average over seeds        
lls_average = np.mean(lls,axis=4)
            
## step sizes
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = np.int(round(T/dt))
step_size = np.zeros((nt,1))
step_size_init = [0.3]
for i in range(nt):
    step_size[i] = min(step_size_init[0],step_size_init[0]/(t_vals[i]**0.7))

t_fin_indices = [10,25,50,75]
N_vals_to_plot = [5,10,50]

from matplotlib.ticker import StrMethodFormatter

for o in range(len(t_fin_indices)):
    color_counter = 0 
    plt.clf()
    for m in range(len(N_vals)):
        
        if N_vals[m] in N_vals_to_plot:
            ## average marginal likelihods over first t time steps
            lls_average_t_average = np.zeros((len(beta_vals)))
            lls_average_weighted_t_average = np.zeros((len(beta_vals)))
            for i in range(len(beta_vals)):
                lls_average_t_average[i] = np.mean(lls_average[0,i,m,0:t_fin_indices[o]])
                lls_average_weighted_t_average[i] = np.mean(step_size[0:t_fin_indices[o]]*lls_average[0,i,m,0:t_fin_indices[o]])
            
            ## find index of maximum
            max_index = np.argmax(lls_average_t_average)
            
            plt.plot(beta_vals[:],1/(T*N_vals[m])*lls_average_t_average,label="N={}".format(N_vals[m]))
            plt.axvline(beta_vals[max_index],linestyle="--",color="C{}".format(color_counter))
            plt.xlabel(r"$\theta_2$")
            plt.ylabel(r"$\frac{1}{t}\mathcal{L}_t(\theta_2)$")
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
            #plt.axvline(beta_true,color="black",linestyle="-")
            #ax.set_zlabel(r'$\tilde{\mathcal{L}}(\theta)$')
            color_counter += 1
    
    plt.legend(loc="upper right")
    filename = 'log_lik_beta_for_papers_different_N_T_{}.pdf'.format(t_fin_indices[o])
    save_plot(fig_directory,filename)
    plt.show()
    
        
    
    
    
## alpha
alpha_true = [0.5]#[0.25,0.5,0.75,1,1.25,1.5,1.75]
beta_true = 0.1
alpha_vals = np.linspace(0.3,0.7,41)

N_vals = [2,5,10,25,50,100]

T=100;
n_runs = 20
seed_vals=np.linspace(1,n_runs,n_runs);
sigma = 1; x0 = 2; dt = 0.1; 
v_func = linear_func;

lls = np.zeros((len(alpha_true),len(alpha_vals),len(N_vals),T+1,n_runs))
        
for l in range(n_runs):
    for m in range(len(N_vals)):
        
        for j in range(len(alpha_true)):
            
            N = N_vals[m];  alpha = alpha_true[j]; 
            Aij = None; Lij = None; Aij_calc = False
            beta = beta_true; 
            xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,Aij=Aij,
                              Lij=Lij,sigma=sigma,x0=x0,dt=dt,seed=np.int(seed_vals[l]),Aij_calc=Aij_calc)
            
            #plt.plot(xt)
            #plt.show()
            
            dxt = np.diff(xt,axis=0)
            
            for i in range(len(alpha_vals)):
                Lij = laplacian_mat(N,mean_field_mat(N,beta_true))
                lls[j,i,m,:,l] = log_lik_lij(Lij = Lij,N=N,t=T,xt=xt,dxt=dxt,v_func=v_func,
                                       alpha = alpha_vals[i],sigma=sigma,dt=dt,
                                       Aij_calc=Aij_calc,marginal=True)[1]
    
    ## for sanity
    print(l)

## save!
os.chdir(results_directory)
filename = 'online_est_sim_log_lik_alpha_plots.pdf'
np.save(filename,lls)
os.chdir(default_directory)

## reopen!
#os.chdir(results_directory)
#filename = 'online_est_sim_log_lik_alpha_plots.npy'
#lls = np.load(filename)
#os.chdir(default_directory) 

## average over seeds        
lls_average = np.mean(lls,axis=4)
            
## step sizes
t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
nt = np.int(round(T/dt))
step_size = np.zeros((nt,1))
step_size_init = [0.3]
for i in range(nt):
    step_size[i] = min(step_size_init[0],step_size_init[0]/(t_vals[i]**0.7))

t_fin_indices = [10,25,50,75]
N_vals_to_plot = [5,10,50]

from matplotlib.ticker import StrMethodFormatter

for o in range(len(t_fin_indices)):
    color_counter = 0 
    plt.clf()
    for m in range(len(N_vals)):
        
        if N_vals[m] in N_vals_to_plot:
            ## average marginal likelihods over first t time steps
            lls_average_t_average = np.zeros((len(alpha_vals)))
            lls_average_weighted_t_average = np.zeros((len(alpha_vals)))
            for i in range(len(alpha_vals)):
                lls_average_t_average[i] = np.mean(lls_average[0,i,m,0:t_fin_indices[o]])
                lls_average_weighted_t_average[i] = np.mean(step_size[0:t_fin_indices[o]]*lls_average[0,i,m,0:t_fin_indices[o]])
            
            ## find index of maximum
            max_index = np.argmax(lls_average_t_average)
            
            plt.plot(alpha_vals[:],1/(T*N_vals[m])*lls_average_t_average,label="N={}".format(N_vals[m]))
            plt.axvline(alpha_vals[max_index],linestyle="--",color="C{}".format(color_counter))
            plt.xlabel(r"$\theta_2$")
            plt.ylabel(r"$\frac{1}{t}\mathcal{L}_t(\theta_1)$")
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
            #plt.axvline(beta_true,color="black",linestyle="-")
            #ax.set_zlabel(r'$\tilde{\mathcal{L}}(\theta)$')
            color_counter += 1
    
    plt.legend(loc="upper right")
    filename = 'log_lik_alpha_for_papers_different_N_T_{}.pdf'.format(t_fin_indices[o])
    save_plot(fig_directory,filename)
    plt.show()
    


    
## beta and alpha
beta_true =  [0.1]
alpha_true = [0.5]

N_vals = [2,5,10,25,50,100]

for m in N_vals:

    for j in alpha_true:
        for k in beta_true:
        
            N = m; T=2000; 
            alpha = j; beta = k
            Aij = None; Lij = None;
            sigma = .1; x0 = np.linspace(0,1,N); dt = 0.1; seed = 1; 
            v_func = linear_func;
            Aij_calc = False
            
            xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,Aij=Aij,
                              Lij=Lij,sigma=sigma,x0=x0,dt=dt,seed=seed,Aij_calc=Aij_calc)
            
            #plt.plot(xt)
            #plt.show()
            
            dxt = np.diff(xt,axis=0)
            
            alpha_vals = np.linspace(-1,2,61)
            beta_vals = np.linspace(-2,2,81)
            lls = np.zeros((len(alpha_vals),len(beta_vals)))
            
            if False:
                for i in range(len(alpha_vals)):
                    for l in range(len(beta_vals)):
                        Lij = laplacian_mat(N,mean_field_mat(N,beta_vals[l]))
                        lls[i,l] = log_lik_lij(Lij = Lij,N=N,t=T,xt=xt,dxt=dxt,v_func=v_func,
                                               alpha = alpha_vals[i],sigma=sigma,dt=dt,Aij_calc=Aij_calc)
                        
                    print (m,i)
            
            
            ## save 
            #os.chdir(results_directory)
            #filename = 'online_est_sim_log_lik_contours_N_{}.pdf'.format(m)
            #np.save(filename,lls)
            #os.chdir(default_directory)
               
            ## load
            #os.chdir(results_directory)
            #filename = 'online_est_sim_log_lik_contours_N_{}.pdf.npy'.format(m)
            #lls = np.load(filename)
            #os.chdir(default_directory) 
    
            max_index = np.unravel_index(np.argmax(lls),lls.shape)
            X,Y = np.meshgrid(alpha_vals,beta_vals)
            plt.contourf(X,Y,1/(N*T)*lls.T,levels=30,cbar=True,cmap=cm.viridis)
            #for l in range(5):
            #    plt.plot(alpha_est_rml[:,i,l],beta_est_rml[:,i,l])
            plt.colorbar()
            plt.axvline(alpha_true,linestyle="--",color="C1")
            #plt.axvline(alpha_vals[max_index[0]],linestyle="--",color="C1")
            plt.axhline(beta_true,linestyle="--",color="C1")
            #plt.axhline(beta_vals[max_index[1]],linestyle="--",color="C1")
            plt.xlabel(r'$\theta_1$')
            plt.ylabel(r'$\theta_2$')
            filename = 'online_est_sim_log_lik_2d_contour_N_{}.pdf'.format(m)
            #save_plot(fig_directory,filename)
            plt.show()
            
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            surf = ax.plot_surface(X, Y, 1/(N*T)*lls.T, rstride=1, cstride=1,
                                   cmap=cm.viridis,edgecolor="none")
            fig.colorbar(surf,ax=ax,shrink=.9,aspect=15,pad=0.1)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            ax.azim = -60
            ax.dist= 10
            ax.elev= 30
            filename = 'online_est_sim_log_lik_3d_contour_N_{}.pdf'.format(m)
            #save_plot(fig_directory,filename)
            #ax.set_zlabel(r'$\tilde{\mathcal{L}}(\theta)$')
            plt.show()


## beta and alpha (MVSDE)
beta_true =  [0.1]
alpha_true = [0.5]

N_vals = [50]#,25,50,100]

for m in N_vals:

    for j in alpha_true:
        for k in beta_true:
        
            N = m; T=1000; 
            alpha = j; beta = k
            sigma = .1; x0 = np.linspace(0,1,N); 
            dt = 0.1; seed = 1; 
            
            xt = mvsde_sim_func(N=N,T=T,alpha=alpha,beta=beta,sigma=sigma,
                                x0=x0,dt=dt,seed=seed)
            
            #plt.plot(xt)
            #plt.show()
            
            dxt = np.diff(xt,axis=0)
            
            alpha_vals = np.linspace(0.2,2,31)
            beta_vals = np.linspace(-2,2,41)
            lls = np.zeros((len(alpha_vals),len(beta_vals)))
            
            if True:
                for i in range(len(alpha_vals)):
                    for l in range(len(beta_vals)):
                        lls[i,l] = log_lik_mvsde(N=N,t=T,xt=xt,dxt=dxt,
                                                 alpha = alpha_vals[i],
                                                 beta = beta_vals[l],
                                                 sigma=sigma,x0=x0,dt=dt)
                        
                    print (m,i)
            
            
            ## save 
            #os.chdir(results_directory)
            #filename = 'online_est_sim_log_lik_contours_N_{}.pdf'.format(m)
            #np.save(filename,lls)
            #os.chdir(default_directory)
               
            ## load
            #os.chdir(results_directory)
            #filename = 'online_est_sim_log_lik_contours_N_{}.pdf.npy'.format(m)
            #lls = np.load(filename)
            #os.chdir(default_directory) 
    
            max_index = np.unravel_index(np.argmax(lls),lls.shape)
            X,Y = np.meshgrid(alpha_vals,beta_vals)
            plt.contourf(X,Y,1/(N*T)*lls.T,levels=30,cbar=True,cmap=cm.viridis)
            #for l in range(5):
            #    plt.plot(alpha_est_rml[:,i,l],beta_est_rml[:,i,l])
            plt.plot(np.mean(alpha_est_rml[:,i,:],axis=1),np.mean(beta_est_rml[:,i,:],axis=1))
            plt.colorbar()
            plt.axvline(alpha_true,linestyle="--",color="C1")
            #plt.axvline(alpha_vals[max_index[0]],linestyle="--",color="C1")
            plt.axhline(beta_true,linestyle="--",color="C1")
            #plt.axhline(beta_vals[max_index[1]],linestyle="--",color="C1")
            plt.xlabel(r'$\theta_1$')
            plt.ylabel(r'$\theta_2$')
            filename = 'online_est_sim_log_lik_2d_contour_N_{}.pdf'.format(m)
            #save_plot(fig_directory,filename)
            plt.show()
            
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            surf = ax.plot_surface(X, Y, 1/(N*T)*lls.T, rstride=1, cstride=1,
                                   cmap=cm.viridis,edgecolor="none")
            fig.colorbar(surf,ax=ax,shrink=.9,aspect=15,pad=0.1)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            ax.azim = -60
            ax.dist= 10
            ax.elev= 30
            filename = 'online_est_sim_log_lik_3d_contour_N_{}.pdf'.format(m)
            #save_plot(fig_directory,filename)
            #ax.set_zlabel(r'$\tilde{\mathcal{L}}(\theta)$')
            plt.show()
            


### 'Bump' interaction function ###

## 'centre' parameter
centre_true = [-0.2,-0.1,0,0.1,0.2,0.3,0.4]

for j in centre_true:
    
    N = 30; T = 20; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.1; 
    x0 = np.linspace(0,5,N); dt = 0.1; seed = 1; v_func = null_func; Aij_calc = True
    Aij_influence_func = influence_func_bump; Aij_calc_func = Aij_calc_func
    Aij_grad_influence_func = influence_func_bump_grad; 
    Aij_grad_calc_func = Aij_grad_calc_func; Aij_scale_param = 2
    
    centre_true = j; width=1; squeeze = .01
    
    x_vals = np.linspace(0,1.5,201)
    y_vals = [influence_func_bump(x,j,width,squeeze) for x in x_vals]
    plt.plot(x_vals,y_vals)
    plt.show()

    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      Aij_scale_param = Aij_scale_param,
                      centre = j, width=width, squeeze=squeeze)
    plt.plot(xt)
    plt.show()

    dxt = np.diff(xt,axis=0)
    

    centre = np.linspace(-0.5*width,0.8,101)
    lls = [0]*len(centre)
    
    grad_indices =  [0]
    ll_grads = np.zeros((len(centre),len(grad_indices)))
    
    for i in range(len(centre)):

        lls[i] = log_lik_lij(Lij = None,N=N,t=T,xt=xt,dxt = dxt,
                             v_func=v_func,alpha=alpha,sigma=sigma,dt=dt,
                             Aij_calc = True, Aij_calc_func = Aij_calc_func,
                             Aij_influence_func = Aij_influence_func, 
                             Aij_scale_param = Aij_scale_param,
                             centre = centre[i], width = width,
                             squeeze = squeeze)
        
        if False:
            ll_grads[i] = log_lik_lij_grad(Lij=None,N=N,t=T,xt=xt,dxt=dxt,
                                           v_func=v_func,alpha=alpha,sigma=sigma,
                                           dt=dt,Aij_calc = True, 
                                           Aij_calc_func = Aij_calc_func,
                                           Aij_influence_func = Aij_influence_func, 
                                           Aij_grad_calc_func = Aij_grad_calc_func, 
                                           Aij_grad_influence_func = Aij_grad_influence_func,
                                           n_param = 3, grad_indices = grad_indices,
                                           Aij_scale_param = Aij_scale_param,
                                           centre = centre[i], width = width, 
                                           squeeze = squeeze)[grad_indices]
    
    ## we will plot the 'right edge' (equivalent to plotting centre, since
    ## the width is fixed)
    
    right_edge = centre+0.5*width
    plt.plot(right_edge,lls)
    #plt.plot(right_edge[1:],np.diff(lls)/np.diff(right_edge))
    #plt.plot(right_edge,ll_grads)
    max_index = lls.index(max(lls))
    plt.axvline(centre[max_index]+0.5*width,linestyle="--",color="C1")
    plt.axvline(j+0.5*width,linestyle="--",color="C2")
    plt.xlabel(r'${\theta}$'); plt.ylabel(r'${\mathcal{L}(\theta)}$')
    filename = 'offline_est_sim_7_{}.pdf'.format(j)
    save_plot(fig_directory,filename)
    plt.show()
    
    #plt.plot(centre,lls)
    #max_index = lls.index(max(lls))
    #plt.axvline(centre[max_index],linestyle="--")
    #plt.axvline(j,linestyle="--")
    #plt.show()
    
    
## some important notes:
    
## when the 'width' parameter is too large and/or the initial conditions
## aren't sufficiently spread out, the 'centre' parameter is no longer 
## identifiable: this is because the particle dynamics are now exactly the same 
## regardless of the value of the 'centre' parameter, and hence the log-lik
## is also the same for different values of this parameter (hence the
## parameter can't be identified)


## the log-likelihood will be identically zero if Lij (or Aij) is identically 
## zero: this will happen when the interaction is zero for all positive 
## distance (or only non-zero  for very very small positive distances)


## there isn't much hope for a gradient descent method, even in the offline 
## case; the gradient of the log-likelihood fluctuates very wildly for 
## 'reasonable' values of $N$ and $T$ (the situation may improve as N and 
## T increase)


## more comments in param_est_interac_sims.py




## multiple 'centre' parameters
n_kernels = 2
n_centres = 1
centre_true = [0,.5]; width=[1]*n_kernels; squeeze = [.01]*n_kernels
x_vals = np.linspace(0,1.5,201)
y_vals = [influence_func_mult_bump(x,centre_true,width,squeeze) for x in x_vals]
plt.plot(x_vals,y_vals)
plt.xlabel(r'$r$'); plt.ylabel(r'${\phi(r)}$')
plt.show()

    

for j in range(n_centres):
    
    N = 40; T = 30; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.1; 
    x0 = np.linspace(0,5,N); dt = 0.1; seed = 1; v_func = null_func; Aij_calc = True
    Aij_influence_func = influence_func_mult_bump; 
    Aij_calc_func = Aij_calc_func
    Aij_grad_influence_func = influence_func_mult_bump_grad; 
    Aij_grad_calc_func = Aij_grad_calc_func;
    Aij_scale_param = 2

    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      Aij_scale_param = Aij_scale_param,
                      centre = centre_true, width=width, squeeze=squeeze)
    plt.plot(xt)
    plt.show()

    dxt = np.diff(xt,axis=0)
    

    centre1 = np.linspace(-0.5*width[0],0.8,21) # for testing likelihood
    centre2 = np.linspace(-0.5*width[0],0.8,21) # for testing likelihood
    #centre1 = np.linspace(0.45,0.55,51) # for testing gradients
    #centre2 = np.linspace(-0.1,0.1,51) # for testing gradients
    centres = np.zeros((len(centre1),2))
    centres[:,0] = centre1; centres[:,1] = centre2
    lls = np.zeros((len(centre1),len(centre2)))
    
    compute_grads = False
    
    ## gradient w.r.t the two centre parameters
    grad_indices =  [0,3]
    ll_grads = np.zeros((len(centre1),len(centre2),len(grad_indices)))
    
    for i in range(len(centre1)):
        for j in range(len(centre2)):

            lls[i,j] = log_lik_lij(Lij = None,N=N,t=T,xt=xt,dxt = dxt,
                                 v_func=v_func,alpha=alpha,sigma=sigma,dt=dt,
                                 Aij_calc = True, Aij_calc_func = Aij_calc_func,
                                 Aij_influence_func = Aij_influence_func, 
                                 Aij_scale_param = Aij_scale_param,
                                 centre = [centres[i,0],centres[j,1]], 
                                 width = width,squeeze = squeeze)
            
            if compute_grads:
                ll_grads[i,j,:] = log_lik_lij_grad(Lij=None,N=N,t=T,xt=xt,dxt=dxt,
                                               v_func=v_func,alpha=alpha,sigma=sigma,
                                               dt=dt,Aij_calc = True, 
                                               Aij_calc_func = Aij_calc_func,
                                               Aij_influence_func = Aij_influence_func, 
                                               Aij_grad_calc_func = Aij_grad_calc_func, 
                                               Aij_grad_influence_func = Aij_grad_influence_func,
                                               n_param = 6, grad_indices = grad_indices,
                                               Aij_scale_param = Aij_scale_param,
                                               centre = [centres[i,0],centres[j,1]],
                                               width = width, squeeze = squeeze)[grad_indices]
                
        print(i)
    
    
    
    ## plots w.r.t to 'right edge' rather than 'centre'             
    right_edge = centres+0.5*width[0]
    x,y = np.meshgrid(right_edge[:,0],right_edge[:,1])
    
    ## plot lls
    
    ## (i) surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x,y,lls)

    
    ## (ii) contour plot
    fig, ax = plt.subplots()
    cf = ax.contourf(x,y,lls,50)
    fig.colorbar(cf, ax=ax)
    max_index = np.argmax(lls)
    max_indices = np.unravel_index(max_index,lls.shape)
    plt.axvline(x[max_indices[0],max_indices[1]],linestyle="--",color="C1")
    plt.axhline(y[max_indices[0],max_indices[1]],linestyle="--",color="C1")
    plt.axhline(centre_true[0]+0.5*width[0],linestyle="--",color="C2")
    plt.axvline(centre_true[1]+0.5*width[0],linestyle="--",color="C2")
    plt.xlabel(r'${\theta}_1$'); plt.ylabel(r'${\theta}_2$')
    filename = 'offline_est_sim_8_{}.pdf'.format(j)
    #save_plot(fig_directory,filename)
    plt.show()
    
    
    ## plot ll_grads
    
    if compute_grads:
        
        ## (i) quiver plot
        fig, ax = plt.subplots()
        ax.quiver(x,y,ll_grads[:,:,0],ll_grads[:,:,1])
        max_index = np.argmax(lls)
        max_indices = np.unravel_index(max_index,lls.shape)
        plt.axhline(centres[max_indices[0],0]+0.5*width[0],linestyle="--",color="C1")
        plt.axvline(centres[max_indices[1],1]+0.5*width[0],linestyle="--",color="C1")
        plt.axhline(centre_true[0]+0.5*width[0],linestyle="--",color="C2")
        plt.axvline(centre_true[1]+0.5*width[0],linestyle="--",color="C2")
        plt.show()
        
        ## (ii) compare to 'empirical' gradient
        ll_grad_est1 = np.diff(lls[:,0])/np.diff(centre1)
        ll_grad_tru1 = ll_grads[:,0,0]
        diff1 = ll_grad_est1 - ll_grad_tru1
        plt.plot(ll_grad_est1); plt.plot(ll_grad_tru1)
        
        ll_grad_est2 = np.diff(lls[0,:])/np.diff(centre2)
        ll_grad_tru2 = ll_grads[0,:,1]
        plt.plot(ll_grad_est2); plt.plot(ll_grad_tru2)
    



### 'Gaussian' interaction function ###

## 'mu' parameter
mu_true = [0.25,0.5,0.75,1.0]

for j in mu_true:

    mu=j; sd = 0.5
    x_vals = np.linspace(0,2,201)
    y_vals = [influence_func_exp(x,mu,sd) for x in x_vals]
    plt.plot(x_vals,y_vals)
    plt.show()
    
    N = 10; T = 5; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.1; 
    x0 = np.linspace(0,5,N); dt = 0.05; seed = 1; v_func = null_func; Aij_calc = True
    Aij_influence_func = influence_func_exp; Aij_calc_func = Aij_calc_func
    Aij_grad_influence_func = influence_func_exp_grad; 
    Aij_grad_calc_func = Aij_grad_calc_func
    
    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      mu=mu, sd=sd)
    
    dxt = np.diff(xt,axis=0)
    #plt.plot(xt)

    mu = np.linspace(0,2,41)
    lls = [0]*len(mu)
    
    grad_indices =  [0]
    ll_grads = np.zeros((len(centre),len(grad_indices)))
    
    for i in range(len(mu)):
    
        lls[i] = log_lik_lij(Lij = None,N=N,t=T,xt=xt,dxt = dxt,
                             v_func=v_func,alpha=alpha,sigma=sigma,dt=dt,
                             Aij_calc = True, Aij_calc_func = Aij_calc_func,
                             Aij_influence_func = Aij_influence_func, 
                             mu = mu[i], sd=sd)
        
        if True:
            ll_grads[i] = log_lik_lij_grad(Lij=None,N=N,t=T,xt=xt,dxt=dxt,
                                           v_func=v_func,alpha=alpha,sigma=sigma,
                                           dt=dt,Aij_calc = True, 
                                           Aij_calc_func = Aij_calc_func,
                                           Aij_influence_func = Aij_influence_func, 
                                           Aij_grad_calc_func = Aij_grad_calc_func, 
                                           Aij_grad_influence_func = Aij_grad_influence_func,
                                           n_param = 3,grad_indices = grad_indices, 
                                           mu = mu[i], sd=sd)[grad_indices]
        
        
    max_index = lls.index(max(lls))
    plt.plot(mu,lls)
    plt.axvline(mu[max_index],linestyle="--")
    plt.axvline(j,linestyle="--")
    plt.show()
    
    
    
## 'sd' parameter
sd_true = [0.25,0.3,0.35,0.4]

for j in sd_true:

    mu=0.5; sd = j
    x_vals = np.linspace(0,2,201)
    y_vals = [influence_func_exp(x,mu,sd) for x in x_vals]
    plt.plot(x_vals,y_vals)
    plt.show()
    
    N = 40; T = 30; alpha = 0; beta = None; Aij = None; Lij = None; sigma = 0.1; 
    x0 = np.linspace(0,1,N); dt = 0.05; seed = 1; v_func = null_func; Aij_calc = True
    Aij_influence_func = influence_func_exp; Aij_calc_func = Aij_calc_func
    
    xt = sde_sim_func(N=N,T=T,v_func=null_func,alpha=alpha,beta = beta,
                      Aij = Aij,Lij = Lij,sigma=sigma,x0=x0,dt=dt,
                      seed=seed,Aij_calc=Aij_calc, 
                      Aij_calc_func = Aij_calc_func,
                      Aij_influence_func = Aij_influence_func,
                      mu=mu, sd=sd)
    
    dxt = np.diff(xt,axis=0)
    #plt.plot(xt)

    sd = np.linspace(0.1,2,81)
    lls = [0]*len(sd)
    for i in range(len(sd)):
    
        lls[i] = log_lik_lij(Lij = None,N=N,t=T,xt=xt,dxt = dxt,
                             v_func=v_func,alpha=alpha,sigma=sigma,dt=dt,
                             Aij_calc = True, Aij_calc_func = Aij_calc_func,
                             Aij_influence_func = Aij_influence_func, 
                             mu = mu, sd=sd[i])
        
    
    max_index = lls.index(max(lls))
    plt.plot(sd,lls)
    plt.axvline(sd[max_index],linestyle="--")
    plt.axvline(j,linestyle="--")
    plt.show()
    















#######################################
#######################################
#######################################
################ OTHER ################
#######################################
#######################################
#######################################

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

#######################################
#######################################
#######################################

