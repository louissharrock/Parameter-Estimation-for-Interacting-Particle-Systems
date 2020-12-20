#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:56:47 2020

@author: ls616
"""

## Interacting Particles SDE (Functions) ##

#################
#### Prelims ####
#################

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
import os
from copy import deepcopy

#################


#######################
#### AUXILIARY FNs ####
#######################

## Save plots
def save_plot(directory,filename):
    os.chdir(directory)
    plt.savefig(filename,bbox_inches='tight')
    os.chdir("/Users/ls616")
    
## Extract upper diagonal matrix (without diagonal entries)
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

#######################



######################
#### MATRIX NORMS ####
######################

## Frobenius Norm
def frob_norm(x):
    return(np.linalg.norm(x))


## L1 norm
def l1_norm(x):
    return(np.sum(abs(x)))

## l2 norm
def l2_norm(x):
    return(np.sum(abs(x)**2))

######################


########################
#### STEP SIZE FUNC ####
########################

def step_size_decrease(T,dt,step_size_init,power,delay=0):
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## values of T
    t_vals = np.linspace(0,T,int(np.round(T/dt))+1)
    
    ## delayed values of T
    t_vals_delay = np.zeros(nt+1)
    for i in range(nt):
       t_vals_delay[i] = max(0,t_vals[i]-delay)
    
    ## step-sizes
    step_sizes = np.zeros(nt)
    for i in range(nt):
        step_sizes[i] = min(abs(step_size_init),abs(step_size_init/(t_vals_delay[i]**power)))
        
    return step_sizes
    
    

########################



#############################
#### POTENTIAL FUNCTIONS ####
#############################

## Gradient of quadratic potential (linear)
def linear_func(x,alpha):
    return(alpha*x) 

def null_func(x,alpha):
    return(0)

#############################
    
    
    
###############################
#### INTERACTION FUNCTIONS ####
###############################
    
## Mean Field Matrix
## - compute Aij matrix for mean field
def mean_field_mat(N,beta):
    mat = beta*1/N*np.ones((N,N))
    #np.fill_diagonal(mat,0)
    return(mat)

## Graph Laplacian Matrix 
## - compute Lij matrix from Aij matrix
def laplacian_mat(N,Aij):
    col_sums = np.zeros(N)
    for i in range(N):
        col_sums[i] = sum(Aij[:,i])
    
    mat = np.diag(col_sums) - Aij
    return(mat)

## Interaction Matrix
## - compute Aij matrix from Lij matrix
## - need to input col sums from original matrix to recover Aij matrix from Lij matrix
def interaction_mat(N,Lij,Aij_colsum):
    mat = - Lij
    for i in range(N):
        mat[i,i] =  Aij_colsum[i] - Lij[i,i]
    return(mat)


## Mean Field Column Sums ##
## - estimate column sums of mean field Aij matrix, using Lij matrix 
def mean_field_col_sums(N,Lij):
    off_diag_average = np.zeros(N)
    for i in range(N):
        off_diag_average[i] = np.mean(np.delete(Lij[:,i],i))
    
    col_sums = -N*off_diag_average
        
    return col_sums


## Zero Diagonal Column Sums ##
## - estimate column sums of mean field Aij matrix, using Lij matrix
def zero_diag_col_sums(N,Lij):
    
    col_sums = np.zeros(N)
    
    for i in range(N):
        col_sums[i] = - np.sum(np.delete(Lij[:,i],i))

    return col_sums


## Tri-Diagonal Interaction matrix ##
def tri_diag_interaction_mat(N,interaction):
    mat = np.diag(N*[0])+np.diag((N-1)*[interaction],1) \
        + np.diag((N-1)*[interaction],-1) \
        + np.diag((N-2)*[interaction],+2) \
        + np.diag((N-2)*[interaction],-2)
    
    mat[0,(N-2):] = mat[(N-2):,0] = interaction
    mat[1,(N-1):] = mat[(N-1):,1] = interaction
    
    return mat


## Degree Matrix ##
def deg_matrix(Aij):
    mat = np.diag(np.sum(Aij,axis=0))
    return(mat)


## Connectivity Matrix ##
def connect_matrix(Aij):
    mat = np.where(Aij>0,1,0)
    return(mat)
            

###############################



#######################
#### SDE SIMULATOR ####
#######################
    
## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> v_func (gradient field function)
## -> alpha (parameter for gradient field function)
## -> beta (paramater for mean field, optional)
## -> Aij (interaction matrix, optional)
## -> Lij (laplacian matrix), optional)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)
    
def sde_sim_func(N=1,T=100,v_func=linear_func,alpha=0.1,beta=None,
                 Aij=None,Lij=None,sigma=1,x0=1,dt=0.1,seed=1):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)

    ## simulate
    if(beta!=None): ## mean field
        for i in range(0,nt):
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - beta*(xt[i,:] - np.mean(xt[i,:]))*dt + sigma*dwt[i,:]
    if(np.any(Aij)): ## Aij form
        for i in range(0,nt):
            for j in range(0,N):
                xt[i+1,j] = xt[i,j] - v_func(xt[i,j],alpha)*dt - np.dot(Aij[:,j],xt[i,j]-xt[i,:])*dt + sigma*dwt[i,j]
    if(np.any(Lij)): ## Lij form
        for i in range(0,nt):
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - np.dot(Lij,xt[i,:])*dt + sigma*dwt[i,:]
        
    return(xt)

#######################



#######################################
#### MEAN FIELD ESIMATOR (OFFLINE) ####
#######################################
    
## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> v_func (gradient field function)
## -> alpha (parameter for gradient field function)
## -> beta (paramater for mean field, optional)
## -> Aij (interaction matrix, optional)
## -> Lij (laplacian matrix), optional)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)
## -> mle (whether to compute the mle or the map)

def mean_field_mle_map(N=10,T=100,v_func=linear_func,alpha=0.1,beta=1,
                         sigma=1,x0=2,dt=0.1,seed=1,mle=True):
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## compute x_t
    xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                      sigma=sigma,x0=x0,dt=dt,seed=seed)
    
    ## compute 'dx_t'
    dxt = np.diff(xt,axis=0)
    
    ## numerator 
    num = np.zeros((nt,N))
    for i in range(0,nt):
        for j in range(0,N):
            num[i,j] = -(xt[i,j] - np.mean(xt[i,:]))*dxt[i,j] - v_func(xt[i,j],alpha)*(xt[i,j]-np.mean(xt[i,:]))*dt
    
    numerator = np.sum(np.cumsum(num[1:,:],axis=0),axis=1) # integrate over time & sum over particles
    
    ## denominator 
    denom = np.zeros((nt,N))
    for i in range(0,nt):
        for j in range(0,N):
            denom[i,j] = (xt[i,j] - np.mean(xt[i,:]))**2*dt
    
    denominator = np.sum(np.cumsum(denom[1:,],axis=0),axis=1) # integrate over time & sum over particles
    
    
    ## compute estimator: 
        
    ## mle
    if(mle):
        estimator = numerator/denominator
    
    ## MAP
    else:
        estimator = (1+numerator)/(1+denominator)
        
        
    ## compute MSE 
    mse = (estimator-beta)**2
    
    
    ## output
    return estimator,mse

#######################################



######################################
#### MEAN FIELD ESIMATOR (ONLINE) ####
######################################

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> v_func (gradient field function)
## -> alpha (parameter for gradient field function)
## -> beta (paramater for mean field, optional)
## -> Aij (interaction matrix, optional)
## -> Lij (laplacian matrix), optional)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)
## -> step_size (step sizes)

def mean_field_rmle(N=10,T=500,v_func=linear_func,alpha=0.1,beta=1,beta0=0,
                   sigma=1,x0=2,dt=0.1,seed=1,step_size=0.01):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## intialise beta_est
    beta_est = np.zeros(nt+1)
    beta_est[0] = beta0
    
    ## step sizes
    if(type(step_size)==float):
        step_size = [step_size]*nt
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
    
    for i in range(nt):
        
        ## simulate sde
        xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - beta*(xt[i,:] - np.mean(xt[i,:]))*dt + sigma*dwt[i,:]
        
        ## update parameter
        beta_est[i+1] = beta_est[i] + step_size[i]*sigma**(-2)*np.sum((-(xt[i,:]-np.mean(xt[i,:])))*(xt[i+1,:]-xt[i,:])-(-(xt[i,:]-np.mean(xt[i,:])))*(-v_func(xt[i,:],alpha)-beta_est[i]*(xt[i,:]-np.mean(xt[i,:])))*dt) # sum over all particles
      
    
    ## compute MSE
    mse = (beta_est - beta)**2
    
    return beta_est,mse 
        
######################################



###################################
#### LOG-LIKELIHOOD (LIJ FORM) ####
###################################

def log_lik_lij(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt):
    
    ## reshape if Lij is input as a vector
    if(Lij.shape == (N*N,)):
        Lij = Lij.reshape(N,N)
    
    ## compute log likelihood
    ll_i = np.zeros(t+1)
    for i in range(t+1):
        ll_i[i] = -1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),dxt[i,:])-1/(2*sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]))*dt
    
    ll = np.sum(ll_i)
    return ll


def log_lik_lij_grad(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt):
    
    ## reshape if Lij is input as a vector
    if(Lij.shape == (N*N,)):
        Lij = Lij.reshape(N,N)
    
    ## intialise Lij_grad
    Lij_grad = np.zeros((N,N,N*N))
    
    for i in range(N):
        for j in range(N):
            Lij_grad[i,j,i*N+j] = 1
            
    
    ll_grad_i = np.zeros((t+1,N*N))
    for i in range(t+1):
        for j in range(N*N):
            ll_grad_i[i,j] = -1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij_grad[:,:,j],xt[i,:]),dxt[i,:])-1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),np.dot(Lij_grad[:,:,j],xt[i,:]))*dt
            
    
    ll_grad = np.sum(ll_grad_i,axis=0)
    return ll_grad
    
###################################


##################################
#### LASSO OBJECTIVE FUNCTION ####
##################################

def lasso_objective(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt,lasso_param):
    
    lasso_obj = -1/t*log_lik_lij(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt) + lasso_param*np.sum(abs(Lij))
    
    return lasso_obj


def lasso_objective_grad(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt,lasso_param):
    
    lasso_obj_grad = -1/t*log_lik_lij_grad(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt) + lasso_param*np.where(Lij>0,1,-1).reshape(N*N)
    
    return lasso_obj_grad

##################################



#################################
#### LIJ ESTIMATOR (OFFLINE) ####
#################################

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> v_func (gradient field function)
## -> alpha (parameter for gradient field function)
## -> beta (paramater for mean field, optional)
## -> Lij (laplacian matrix, optional)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)
## -> ridge_param (ridge regression parameter)
## -> mle_ridge (boolean: should we compute the mle or ridge estimator)
## -> lasso (boolean: should we compute the lasso estimator)

def Lij_mle_ridge_lasso(N=2,T=1000,v_func=linear_func,alpha=0.1,beta=0.1,
                        Lij = None, sigma=1,x0=2,dt=0.1,seed=1, ridge_param = 0,
                        mle_ridge = True, lasso = False, lasso_param = 0.01,
                        Lij_init = None, lasso_t_vals = None):
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## compute x_t
    xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta = beta,
                      Lij=Lij,sigma=sigma,x0=x0,dt=dt,seed=seed)
    
    ## compute 'dx_t'
    dxt = np.diff(xt,axis=0)
    
    ## compute mle or ridge estimator
    if(mle_ridge):
        
        ## 'numerator'
        num = np.zeros((nt,N,N))
        for i in range(nt):
            num[i] = np.dot(dxt[i,:].reshape(N,1),xt[i,:].reshape(1,N)) + np.dot(v_func(xt[i,:],alpha).reshape(N,1),xt[i,:].reshape(1,N))*dt
            
        numerator = np.cumsum(num[1:,:,:],axis=0)
        
        ## 'denominator'
        denom = np.zeros((nt,N,N))
        for i in range(nt):
            denom[i] = np.dot(xt[i,:].reshape(N,1),xt[i,:].reshape(1,N))*dt
        
        denominator = np.linalg.inv(np.cumsum(denom[1:,:,:],axis=0) + ridge_param)
        
        ## estimator
        estimator = np.zeros((nt-1,N,N))
        for i in range(nt-1):
            estimator[i,:,:] = -np.dot(numerator[i,:,:],denominator[i,:,:])
            
            ## symmetrise
            estimator[i,:,:] = 0.5*(estimator[i,:,:]+np.transpose(estimator[i,:,:]))
           
            ## ensure off-diagonals are non-positive
            pos_indices = estimator[i,:,:]>0
            np.fill_diagonal(pos_indices,False)
            estimator[i,pos_indices]=0
            
            ## diagonal constraints
            for j in range(N):
                delta = np.sum(estimator[i,:,j])
                estimator[i,j,j] = estimator[i,j,j] - delta
            
            #for j in range(N):
            #       delta = np.sum(estimator[i,:,j])
            #       estimator[i,j,j:] = estimator[i,j,j:] - delta/(N-j)
            #       estimator[i,j:,j] = estimator[i,j,j:]
        
    ## compute lasso estimator
    ## this is quite costly, so we only compute at specified times
    if(lasso):
        estimator = np.zeros((len(lasso_t_vals),N,N))
        
        
        for i in range(len(lasso_t_vals)):
            
            nt_i = int(np.round(lasso_t_vals[i]/dt))
            xt_i = xt[0:(nt_i+1),:]
            dxt_i = dxt[0:(nt_i),:]
            
            estimator[i,:,:] = minimize(fun=lasso_objective,x0 = Lij_init.reshape(N*N,),args = (N,nt_i-1,xt_i,dxt_i,v_func,alpha,sigma,dt,lasso_param),jac=lasso_objective_grad).x.reshape(N,N)
        
            ## symmetrise
            estimator[i,:,:] = 0.5*(estimator[i,:,:]+np.transpose(estimator[i,:,:]))
           
            ## ensure off-diagonals are non-positive
            pos_indices = estimator[i,:,:]>0
            np.fill_diagonal(pos_indices,False)
            estimator[i,pos_indices]=0
            
            ## diagonal constraints
            for j in range(N):
                 delta = np.sum(estimator[i,:,j])
                 estimator[i,j,j] = estimator[i,j,j] - delta
            
            #for j in range(N):
            #       delta = np.sum(estimator[i,:,j])
            #       estimator[i,j,j:] = estimator[i,j,j:] - delta/(N-j)
            #       estimator[i,j:,j] = estimator[i,j,j:]
                   
        
    return(estimator)
    

##################################



################################
#### LIJ ESTIMATOR (ONLINE) ####
################################

def Lij_rmle(N=2,T=1000,v_func=linear_func,alpha = 0.1, Lij = None, 
             Lij_init = None, sigma=1,x0=2,dt=0.1,seed=1,step_size=0.01,
             method = "fast"):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## intialise beta_est
    Lij_est= np.zeros((nt+1,N,N))
    Lij_est[0,:,:] = Lij_init
    Lij_est_output = deepcopy(Lij_est)
    
    ## step sizes
    if(type(step_size)==float):
        step_size = [step_size]*nt
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
    
    if (method=="slow"):
        ## Lij gradient (all elements individually)
        Lij_grad = np.zeros((N,N,N*N))
        for i in range(N):
            for j in range(N):
                Lij_grad[i,j,i*N+j] = 1
        
        for i in range(0,nt):
        
            ## simulate sde
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - np.dot(Lij,xt[i,:])*dt + sigma*dwt[i,:]
            
            ## update parameters (all elements individually)
            Lij_est[i+1,:,:] = Lij_est[i,:,:]
            for j in range(N):
                for k in range(0,N):
                    Lij_est[i+1,j,k] = Lij_est[i,j,k] + step_size[i]*sigma**(-2)*(np.dot(-np.dot(Lij_grad[:,:,j*N+k],xt[i,:]),(xt[i+1,:]-xt[i,:]))-np.dot((-np.dot(Lij_grad[:,:,j*N+k],xt[i,:])),(-v_func(xt[i,:],alpha)-np.dot(Lij_est[i,:,:],xt[i,:])))*dt)       
            
            
            ## enforce constraints (output only)
            Lij_est_output[i+1,:,:] = deepcopy(Lij_est[i+1,:,:])
            
            ## symmetrise
            Lij_est_output[i+1,:,:] = 0.5*(Lij_est_output[i+1,:,:]+Lij_est_output[i+1,:,:].T)
            
            ## non-positive off lead-diagonal
            pos_indices = Lij_est_output[i+1,:,:]>0
            np.fill_diagonal(pos_indices,False)
            Lij_est_output[i+1,pos_indices]=0
            
            ## diagonal constraints
            for j in range(N):
                    delta = np.sum(Lij_est_output[i+1,:,j])
                    Lij_est_output[i+1,j,j:] = Lij_est_output[i+1,j,j:] - delta/(N-j)
                    Lij_est_output[i+1,j:,j] = Lij_est_output[i+1,j,j:]
                   
            #for j in range(N):
            #     delta = np.sum(Lij_est_output[i+1,:,j])
            #     Lij_est_output[i,j,j] = Lij_est_output[i+1,j,j] - delta
                 
            
            if ((i*dt)%10==0): print(i*dt)
    
    
    
    ## faster method (use symmetric structure)
    if (method=="med"):
        
        ## lij gradient
        Lij_grad = np.zeros((N,N,np.int(N*(N+1)/2)))
        count = 0
        for j in range(N):
            for k in range(j,N):
                #print(count+k)
                Lij_grad[j,k,count+k] = 1
                Lij_grad[k,j,count+k] = 1
            count += N-j-1
        
        ## iterate
        for i in range(0,nt):
        
            ## simulate sde
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - np.dot(Lij,xt[i,:])*dt + sigma*dwt[i,:]
            
            ## update parameters
            Lij_est[i+1,:,:] = Lij_est[i,:,:]
            
            count = 0
            
            LdotX = np.dot(Lij_est[i,:,:],xt[i,:])
            vX = v_func(xt[i,:],alpha)
            
            log_lik_grads = np.zeros(np.int(N*(N+1)/2))
            for j in range(N):
                for k in range(j,N):
                    log_lik_grads[count+k] = sigma**(-2)*(np.dot(-np.dot(Lij_grad[:,:,count+k],xt[i,:]),(xt[i+1,:]-xt[i,:]))-np.dot((-np.dot(Lij_grad[:,:,count+k],xt[i,:])),(-vX-LdotX))*dt)
                count += N-j-1
            
            log_lik_grad_mat = np.zeros((N,N))
            for j in range(np.int(N*(N+1)/2)):
                log_lik_grad_mat+=Lij_grad[:,:,j]*log_lik_grads[j]

            ## update
            Lij_est[i+1,:,:] = Lij_est[i,:,:] + step_size[i]*log_lik_grad_mat
            
            
            ## enforce constraints (output only)
            Lij_est_output[i+1,:,:] = deepcopy(Lij_est[i+1,:,:])
            
            ## non-positive off lead-diagonal
            pos_indices = Lij_est_output[i+1,:,:]>0
            np.fill_diagonal(pos_indices,False)
            Lij_est_output[i+1,pos_indices]=0
            
            ## diagonal constraints
            for j in range(N):
                    delta = np.sum(Lij_est_output[i+1,:,j])
                    Lij_est_output[i+1,j,j:] = Lij_est_output[i+1,j,j:] - delta/(N-j)
                    Lij_est_output[i+1,j:,j] = Lij_est_output[i+1,j,j:]
                   
            #for j in range(N):
            #     delta = np.sum(Lij_est_output[i+1,:,j])
            #     Lij_est_output[i,j,j] = Lij_est_output[i+1,j,j] - delta
                 
            if ((i*dt)%10==0): print(i*dt)

    
    
    ## fastest method (use symmetric & diagonal structure)
    if (method=="fast"):

        ## lij gradient 
        Lij_grad = np.zeros((N,N,np.int(N*(N-1)/2)))
        count = 0
        for j in range(N):
            for k in range(j+1,N):
                #print(count+k-1)
                Lij_grad[j,k,count+k-1] = 1
                Lij_grad[k,j,count+k-1] = 1
                Lij_grad[j,j,count+k-1] = -1
                Lij_grad[k,k,count+k-1] = -1
            count += N-j-2
        
        ## iterate 
        for i in range(0,nt):
            
            ## simulate sde
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - np.dot(Lij,xt[i,:])*dt + sigma*dwt[i,:]
                
            ## update parameters 
            Lij_est[i+1,:,:] = Lij_est[i,:,:]
            
            count = 0
            
            LdotX = np.dot(Lij_est[i,:,:],xt[i,:])
            vX = v_func(xt[i,:],alpha)
            
            log_lik_grads = np.zeros(np.int(N*(N-1)/2))
            for j in range(N):
                for k in range(j+1,N):
                    log_lik_grads[count+k-1] = sigma**(-2)*(np.dot(-np.dot(Lij_grad[:,:,count+k-1],xt[i,:]),(xt[i+1,:]-xt[i,:]))-np.dot((-np.dot(Lij_grad[:,:,count+k-1],xt[i,:])),(-vX-LdotX))*dt)
                count += N-j-2
            
            log_lik_grad_mat = np.zeros((N,N))
            for j in range(np.int(N*(N-1)/2)):
                log_lik_grad_mat+= Lij_grad[:,:,j]*log_lik_grads[j]
            
            
            ## update 
            Lij_est[i+1,:,:] = Lij_est[i,:,:] + step_size[i]*log_lik_grad_mat
            
            
            ## enforce constraints (output only)
            Lij_est_output[i+1,:,:] = deepcopy(Lij_est[i+1,:,:])
            
            ## non-positive off lead-diagonal
            pos_indices = Lij_est_output[i+1,:,:]>0
            np.fill_diagonal(pos_indices,False)
            Lij_est_output[i+1,pos_indices]=0
            
            ## reinforce diagonal constraints
            for j in range(N):
                    delta = np.sum(Lij_est_output[i+1,:,j])
                    Lij_est_output[i+1,j,j:] = Lij_est_output[i+1,j,j:] - delta/(N-j)
                    Lij_est_output[i+1,j:,j] = Lij_est_output[i+1,j,j:]
                   
            #for j in range(N):
            #     delta = np.sum(Lij_est_output[i+1,:,j])
            #     Lij_est_output[i,j,j] = Lij_est_output[i+1,j,j] - delta
            
            
            if ((i*dt)%10==0): print(i*dt)
       
        
    return Lij_est_output
################################





