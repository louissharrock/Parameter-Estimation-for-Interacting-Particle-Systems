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
from scipy.optimize import minimize, minimize_scalar
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
    plt.savefig(filename,bbox_inches='tight',dpi=300)
    #os.chdir("/Users/ls616")
    
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

## Gradient of linear potential (null)
def null_func(x,alpha):
    return(0)

## Gradient of bi-stable Landua potential
def landau_func(x,alpha):
    return alpha*(x**3 - x)


#############################
    
    
    
###############################
#### INTERACTION FUNCTIONS ####
###############################
    
## Mean Field Matrix ##
## - compute Aij matrix for mean field
def mean_field_mat(N,beta):
    mat = beta*1/N*np.ones((N,N))
    #np.fill_diagonal(mat,0)
    return(mat)

## Graph Laplacian Matrix ##
## - compute Lij matrix from Aij matrix
def laplacian_mat(N,Aij):
    row_sums = np.zeros(N)
    for i in range(N):
        row_sums[i] = sum(Aij[i,:])
    
    mat = np.diag(row_sums) - Aij
    return(mat)

## Interaction Matrix ##
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


## Gaussian Interaction Kernel ##
def influence_func_exp(r,mu,sd):
    return 1/(np.sqrt(2*np.pi*sd**2))*np.exp(-(r-mu)**2/(2*sd**2))

## Gaussian Interaction Kernel Gradient ##
def influence_func_exp_grad(r,mu,sd):
    
    ## initialise
    grads = [0,0]
    
    grads[0] = (r-mu)/(sd**2)*1/(np.sqrt(2*np.pi*sd**2))*np.exp(-(r-mu)**2/(2*sd**2))
    grads[1] = (-0.5/(sd**2)+0.5/(sd**4)*(r-mu)**2)*1/(np.sqrt(2*np.pi*sd**2))*np.exp(-(r-mu)**2/(2*sd**2))
    
    return grads


## Bump Interaction Kernel ##
def influence_func_bump(r, centre=0.5, width=1, squeeze=1):
    if(-width/2 + centre < r < width/2 + centre):
        return np.exp(-squeeze/(1-(2/width*(r-centre))**2))
    else:
        return 0
    

## Multiple Bump Interaction Kernel ##
def influence_func_mult_bump(r,centre,width,squeeze):
    
    ## Check equal number of each parameters
    if not len(centre)==len(width)==len(squeeze):
        return "Error: the number of each of the parameters must be equal"
    
    else:
        n_param = len(centre)
        kernels = n_param*[0]
        for i in range(n_param):
            if(-width[i]/2+centre[i]<r<width[i]/2+centre[i]):
                kernels[i] =np.exp(-squeeze[i]/(1-(2/width[i]*(r-centre[i]))**2))
            
    return sum(kernels)
    
    
## Bump Interaction Kernel Gradient ##
def influence_func_bump_grad(r,centre=0.5,width=1,squeeze=1):
    
    ## initialise
    grads = [0,0,0]
    
    if(-width/2+centre<r<width/2+centre):
        grads[0] = 8*squeeze/(width**2)*(r-centre)*1/((1-(2/width*(r-centre))**2)**2)*np.exp(-squeeze/(1-(2/width*(r-centre))**2))
        grads[1] = 8*squeeze/(width**3)*(r-centre)**2*1/((1-(2/width*(r-centre))**2)**2)*np.exp(-squeeze/(1-(2/width*(r-centre))**2))
        grads[2] = -1/(1-(2/width*(r-centre))**2)*np.exp(-squeeze/(1-(2/width*(r-centre))**2))
        
    return grads


## Multiple Bump Interaction Kernel Gradient ##
def influence_func_mult_bump_grad(r,centre,width,squeeze):
    
    ## Check equal numbers of parameters 
    if not len(centre)==len(width)==len(squeeze):
        return "Error: the number of each of the parameters must be equal"
    
    else:
        n_param = len(centre)
        
        ## intialise grads, stored as vector
        ## [centre_1,width_1,squeeze_1,centre_2,width_2,squeeze_2,...]
        
        grads = [0]*n_param*3
        
        for i in range(n_param):
        
            if(-width[i]/2+centre[i]<r<width[i]/2+centre[i]):
                grads[i*3+0] = 8*squeeze[i]/(width[i]**2)*(r-centre[i])*1/((1-(2/width[i]*(r-centre[i]))**2)**2)*np.exp(-squeeze[i]/(1-(2/width[i]*(r-centre[i]))**2))
                grads[i*3+1] = 8*squeeze[i]/(width[i]**3)*(r-centre[i])**2*1/((1-(2/width[i]*(r-centre[i]))**2)**2)*np.exp(-squeeze[i]/(1-(2/width[i]*(r-centre[i]))**2))
                grads[i*3+2] = -1/(1-(2/width[i]*(r-centre[i]))**2)*np.exp(-squeeze[i]/(1-(2/width[i]*(r-centre[i]))**2))
        
    return grads
        
    
    
## Indicator Interaction Kernel ##    
def influence_func_indicator(r,up,mid,strength_up,strength_low):
    if (0<=r<mid):
        return strength_low*r
    if (mid<=r<=up):
        return strength_up*r
    else:
        return 0


## Interaction Matrix ##
## - compute Aij matrix, using an interaction function
def Aij_calc_func(x,kernel_func,Aij_scale_param = 1, **kwargs):
    
    ## compute N
    N = len(x)
    
    ## initialise Aij
    Aij = np.diag(N*[0.1])
    Aij_tmp = np.diag(N*[0.1])
    
    ## compute Aij entries (unnormalised)
    for j in range(0,N):
        for k in range(N):
            dist = abs(x[j]-x[k])
            Aij_tmp[j,k] = kernel_func(r=dist,**kwargs)
           
    ## compute Aij entries (normalised)
    row_sums = [0]*N
    
    for j in range(0,N):
        row_sums[j] = np.sum(Aij_tmp[j,:])
        if(row_sums[j]==0):
            row_sums[j]=1
    for j in range(N):
        for k in range(N):
            Aij[j,k] = Aij_scale_param*Aij_tmp[j,k]/row_sums[j]
            Aij[j,k] = Aij_scale_param*Aij_tmp[j,k]/N
            
    ## output
    return Aij


## Interaction Matrix Gradient ##
## - compute gradient of Aij matrix, using an interaction function

def Aij_grad_calc_func(x,kernel_func,kernel_func_grad,n_param,
                       grad_indices = None, Aij_scale_param = 1, **kwargs):
    
    ## compute N
    N = len(x)
    
    ## grad indices
    if grad_indices is None:
        grad_indices = list(range(n_param))
    
    ## initialise Aij_grad
    Aij_tmp = np.zeros((N,N))
    Aij_grad_tmp = np.zeros((N,N,n_param))
    Aij_grad = np.zeros((N,N,n_param))
    
    ## compute Aij entries (unnormalised)
    for j in range(N):
        for k in range(N):
            dist = abs(x[j] - x[k])
            Aij_tmp[j,k] = kernel_func(r=dist,**kwargs)
    
    for i in grad_indices:
        
        ## compute Aij_grad entries (unnormalised)
        for j in range(N):
            for k in range(N):
                dist = abs(x[j] - x[k])
                Aij_grad_tmp[j,k,i] = kernel_func_grad(r=dist,**kwargs)[i]
         
    
        ## compute Aij_grad entries (normalised)
        row_sums = [0]*N
        grad_row_sums = [0]*N
        
        for j in range(N):
            row_sums[j] = np.sum(Aij_tmp[j,:])
            grad_row_sums[j] = np.sum(Aij_grad_tmp[j,:,i])
            if(abs(row_sums[j])<0.001):
                row_sums[j]=1
        for j in range(N):
            for k in range(N):
                Aij_grad[j,k,i] = Aij_scale_param*(Aij_grad_tmp[j,k,i]*row_sums[j]-Aij_tmp[j,k]*grad_row_sums[j])/(row_sums[j]**2)
                Aij_grad[j,k,i] = Aij_scale_param*Aij_grad_tmp[j,k,i]/N
        
    ## output 
    return Aij_grad
        

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
                 Aij=None,Lij=None,sigma=1,x0=1,dt=0.1,seed=1,
                 Aij_calc = False, Aij_calc_func = None,
                 Aij_influence_func = None, Aij_scale_param = 1,
                 kuramoto = False, **kwargs):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    xt_test = np.zeros((nt+1,N))
    xt_test[0,:] = x0
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
    
    ## simulate
    
    ## parameters
    if type(alpha) is float or type(alpha) is np.float64 or type(alpha) is int:
        alpha = [alpha]*nt
        
    if type(beta) is float or type(beta) is np.float64 or type(beta) is int:
        beta = [beta]*nt

    ## if in simple mean field form
    if(beta!=None): 
        if not kuramoto:
            for i in range(0,nt):
                xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha[i])*dt - beta[i]*(xt[i,:] - np.mean(xt[i,:]))*dt + sigma*dwt[i,:]
        else:
            for i in range(0,nt):
                for j in range(N):
                    xt[i+1,j] = xt[i,j] - v_func(xt[i,j],alpha[i])*dt - beta[i]/N*np.sum(np.sin(xt[i,j]-xt[i,:]))*dt + sigma*dwt[i,j]
                while np.any(xt[i+1,:] > + np.pi) or np.any(xt[i+1,:] < - np.pi):
                    xt[i+1,np.where(xt[i+1,:] > +np.pi)] -= 2.*np.pi
                    xt[i+1,np.where(xt[i+1,:] < -np.pi)] += 2.*np.pi
    ## if in Aij form
    if(np.any(Aij) or Aij_calc): 
        for i in range(0,nt):
            
            ## compute Aij if necessary
            if(Aij_calc):
                Aij = Aij_calc_func(x=xt[i,:],kernel_func=Aij_influence_func,
                                    Aij_scale_param = Aij_scale_param, **kwargs)
            
            ## simulate
            for j in range(0,N):
                xt[i+1,j] = xt[i,j] - v_func(xt[i,j],alpha[i])*dt - np.dot(Aij[j,:],xt[i,j]-xt[i,:])*dt + sigma*dwt[i,j]
                
    ## if in Lij form           
    if(np.any(Lij)): 
        for i in range(0,nt):
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha[i])*dt - np.dot(Lij,xt[i,:])*dt + sigma*dwt[i,:]
        
    return xt

#######################


#########################
#### MVSDE SIMULATOR ####
#########################
    
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
    
def mvsde_sim_func(N=1,T=100,alpha=0.5,beta=0.1,sigma=1,x0=1,dt=0.1,seed=1):
    
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

    for i in range(0,nt):
        xt[i+1,:] = xt[i,:] - alpha*xt[i,:]*dt - beta*(xt[i,:] - x0*np.exp(-alpha*i*dt))*dt + sigma*dwt[i,:]
        
    return xt

#########################



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
                         sigma=1,x0=2,dt=0.1,seed=1,mle=True,
                         est_beta = True, est_alpha = False,
                         kuramoto = False):
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## compute x_t
    xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta=beta,
                      sigma=sigma,x0=x0,dt=dt,seed=seed, kuramoto = kuramoto)
    
    ## compute 'dx_t'
    dxt = np.diff(xt,axis=0)
    
    
    ## estimate both alpha and beta
    if est_beta and est_alpha:
            
        ## numerator for alpha
        num_alpha = np.zeros((nt,N))
        for i in range(0,nt):
            for j in range(0,N):
                num_alpha[i,j] = (xt[i,j] - np.mean(xt[i,:]))*dxt[i,j] - xt[i,j]*dxt[i,j]

        numerator_alpha = np.sum(np.cumsum(num_alpha[1:,:],axis=0),axis=1) # integrate over time & sum over particles
        
        ## denominator for beta
        denom_alpha = np.zeros((nt,N))
        for i in range(0,nt):
            for j in range(0,N):
                denom_alpha[i,j] = xt[i,j]**2*dt - (xt[i,j] - np.mean(xt[i,:]))**2*dt 

        denominator_alpha = np.sum(np.cumsum(denom_alpha[1:,],axis=0),axis=1) # integrate over time & sum over particles
        
        
        ## numerator for beta
        a = np.zeros((nt,N))
        b = np.zeros((nt,N))
        c = np.zeros((nt,N))
        d = np.zeros((nt,N))
        for i in range(0,nt):
            for j in range(0,N):
                a[i,j] = xt[i,j]**2*dt
                b[i,j] = (xt[i,j] - np.mean(xt[i,:]))*dxt[i,j]
                c[i,j] = (xt[i,j] - np.mean(xt[i,:]))**2*dt
                d[i,j] = xt[i,j]*dxt[i,j]
        
        A = np.sum(np.cumsum(a[1:,:],axis=0),axis=1)
        B = np.sum(np.cumsum(b[1:,:],axis=0),axis=1)
        C = np.sum(np.cumsum(c[1:,:],axis=0),axis=1)
        D = np.sum(np.cumsum(d[1:,:],axis=0),axis=1)
        
        numerator_beta = A*B-C*D
        denominator_beta = C*(C-A)
        
        numerator = np.array([numerator_alpha,numerator_beta])
        denominator = np.array([denominator_alpha,denominator_beta])
        
        
    else:
        
        ## estimate just beta
        if est_beta:
            
            ## numerator 
            num = np.zeros((nt,N))
            if not kuramoto:
                for i in range(0,nt):
                    for j in range(0,N):
                        num[i,j] = -(xt[i,j] - np.mean(xt[i,:]))*dxt[i,j] - v_func(xt[i,j],alpha)*(xt[i,j]-np.mean(xt[i,:]))*dt
                
            else:
                for i in range(0,nt):
                    for j in range(0,N):
                        num[i,j] = -1/N*np.sum(np.sin(xt[i,j]-xt[i,:]))*dxt[i,j] - v_func(xt[i,j],alpha)*1/N*np.sum(np.sin(xt[i,j]-xt[i,:]))*dt
                        
                        
            numerator = np.sum(np.cumsum(num[1:,:],axis=0),axis=1) # integrate over time & sum over particles
            
            ## denominator 
            denom = np.zeros((nt,N))
            if not kuramoto:
                for i in range(0,nt):
                    for j in range(0,N):
                        denom[i,j] = (xt[i,j] - np.mean(xt[i,:]))**2*dt
                        
            else:        
                for i in range(0,nt):
                    for j in range(0,N):
                        denom[i,j] = (1/N*np.sum(np.sin(xt[i,j]-xt[i,:])))**2*dt
    
            denominator = np.sum(np.cumsum(denom[1:,],axis=0),axis=1) # integrate over time & sum over particles

    
        ## estimate just alpha
        if est_alpha:
            
            ## numerator
            num = np.zeros((nt,N))
            for i in range(0,nt):
                for j in range(0,N):
                    num[i,j] = -xt[i,j]*dxt[i,j] - beta*xt[i,j]*(xt[i,j] - np.mean(xt[i,:]))*dt
            
            numerator = np.sum(np.cumsum(num[1:,:],axis=0),axis=1) # integrate over time & sum over particles
                    
            denom = np.zeros((nt,N))
            for i in range(0,nt):
                for j in range(0,N):
                    denom[i,j] = xt[i,j]**2*dt
            
            denominator = np.sum(np.cumsum(denom[1:,],axis=0),axis=1) # integrate over time & sum over particles
        
    
    ## compute estimator: 
        
    ## mle
    if(mle):
        estimator = numerator/denominator
    
    ## MAP
    else:
        estimator = (1+numerator)/(1+denominator)
        
    
    
    ## output
    return estimator

#######################################



######################################
#### MEAN FIELD ESIMATOR (ONLINE) ####
######################################

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> v_func (gradient field function: must be linear for est. alpha)
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
                   sigma=1,x0=2,dt=0.1,seed=1,step_size=0.01,est_beta = True,
                   est_alpha = False,alpha0=None,norm=True,kuramoto=False):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## parameters
    if type(alpha) is int or type(alpha) is float or type(alpha) is np.float64:
        alpha = [alpha]*nt
        
    if type(beta) is int or type(beta) is float or type(beta) is np.float64:
        beta = [beta]*nt
    
    ## intialise beta_est
    beta_est = np.zeros(nt+1)
    if est_beta:
        beta_est[0] = beta0
    else:
        beta_est[0] = beta
    
    ## initialise alpha_est
    alpha_est = np.zeros(nt+1)
    if est_alpha:
        alpha_est[0] = alpha0
    else:
        alpha_est[0] = alpha[0]
    
    ## step sizes
    if(type(step_size)==float):
        step_size = step_size*np.ones((nt,2))
        
    if(type(step_size)==list):
        step_size_tmp = step_size
        step_size = np.ones((nt,2))
        step_size[:,0] = step_size_tmp[0]*step_size[:,0]
        step_size[:,1] = step_size_tmp[1]*step_size[:,1]
        
    
    if(len(step_size.shape)==1):
        step_size_tmp = step_size
        step_size = np.zeros((nt,2))
        step_size[:,0] = step_size_tmp
        step_size[:,1] = step_size_tmp
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
    
    for i in range(nt):
        
        if not kuramoto:
            ## simulate sde
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha[i])*dt - beta[i]*(xt[i,:] - np.mean(xt[i,:]))*dt + sigma*dwt[i,:]
        
        else:
            for j in range(N):
                xt[i+1,j] = xt[i,j] - v_func(xt[i,j],alpha[i])*dt - beta[i]/N*np.sum(np.sin(xt[i,j]-xt[i,:]))*dt + sigma*dwt[i,j]
            
        ## update parameters
        if (est_alpha):
            if norm:
                alpha_est[i+1] = alpha_est[i] + step_size[i,0]*1/N*sigma**(-2)*np.sum(-xt[i,:]*(xt[i+1,:]-xt[i,:])-(-xt[i,:]*(-v_func(xt[i,:],alpha_est[i])-beta_est[i]*(xt[i,:]-np.mean(xt[i,:]))))*dt)
            else:
                alpha_est[i+1] = alpha_est[i] + step_size[i,0]*sigma**(-2)*(-xt[i,0]*(xt[i+1,0]-xt[i,0])-(-xt[i,0]*(-v_func(xt[i,0],alpha_est[i])-beta_est[i]*(xt[i,0]-np.mean(xt[i,:]))))*dt)
        else:
            alpha_est[i+1] = alpha_est[i]
            
        if (est_beta):
            if not kuramoto:
                if norm:
                    beta_est[i+1] = beta_est[i] + step_size[i,1]*1/N*sigma**(-2)*np.sum((-(xt[i,:]-np.mean(xt[i,:])))*(xt[i+1,:]-xt[i,:])-(-(xt[i,:]-np.mean(xt[i,:])))*(-v_func(xt[i,:],alpha_est[i+1])-beta_est[i]*(xt[i,:]-np.mean(xt[i,:])))*dt) # sum over all particles
                else:
                    beta_est[i+1] = beta_est[i] + step_size[i,1]*sigma**(-2)*((-(xt[i,0]-np.mean(xt[i,])))*(xt[i+1,0]-xt[i,0])-(-(xt[i,0]-np.mean(xt[i,:])))*(-v_func(xt[i,0],alpha_est[i+1])-beta_est[i]*(xt[i,0]-np.mean(xt[i,:])))*dt)
            else:
                if norm:
                    grad=0
                    for j in range(N):
                        grad += 1/N*sigma**(-2)*((-1/N*np.sum(np.sin(xt[i,j]-xt[i,:])))*(xt[i+1,j]-xt[i,j])-(-1/N*np.sum(np.sin(xt[i,j]-xt[i,:])))*(-v_func(xt[i,j],alpha_est[i+1])-beta_est[i]*(1/N*np.sum(np.sin(xt[i,j]-xt[i,:]))))*dt)
                    beta_est[i+1] = beta_est[i] + step_size[i,1]*grad
                        
                else:
                    beta_est[i+1] = beta_est[i] + step_size[i,1]*sigma**(-2)*((-1/N*np.sum(np.sin(xt[i,0]-xt[i,:])))*(xt[i+1,0]-xt[i,0])-(-1/N*np.sum(np.sin(xt[i,0]-xt[i,:])))*(-v_func(xt[i,0],alpha_est[i+1])-beta_est[i]*(1/N*np.sum(np.sin(xt[i,0]-xt[i,:]))))*dt)
        else:
            beta_est[i+1] = beta_est[i]
    
        if kuramoto:
            while np.any(xt[i+1,:] > + np.pi) or np.any(xt[i+1,:] < - np.pi):
                xt[i+1,np.where(xt[i+1,:] > +np.pi)] -= 2.*np.pi
                xt[i+1,np.where(xt[i+1,:] < -np.pi)] += 2.*np.pi
                
    return beta_est,alpha_est 
        
######################################


##################################
#### MVSDE ESTIMATOR (ONLINE) ####
##################################

## just for linear model ##

def mvsde_rmle(N=10,T=500,alpha=0.1,beta=1,alpha0=0,beta0=0,sigma=1,x0=2,dt=0.1,
               seed=1,step_size=[0.1,0.1],est_beta = True, est_alpha = True,
               sim_mvsde = True, sim_particle = False, norm = True):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## intialise beta_est
    beta_est = np.zeros(nt+1)
    if est_beta:
        beta_est[0] = beta0
    else:
        beta_est[0] = beta
    
    ## initialise alpha_est
    alpha_est = np.zeros(nt+1)
    if est_alpha:
        alpha_est[0] = alpha0
    else:
        alpha_est[0] = alpha
    
    ## step sizes
    
    ## if a single number given
    if(type(step_size)==float):
        step_size = step_size*np.ones((nt,2))
    
    ## if a list of two numbers given
    if(type(step_size)==list):
        step_size_tmp = step_size
        step_size = np.ones((nt,2))
        step_size[:,0] = step_size_tmp[0]*step_size[:,0]
        step_size[:,1] = step_size_tmp[1]*step_size[:,1]
        
    ## if step sizes only given for one parameter
    if(len(step_size.shape)==1):
        step_size_tmp = step_size
        step_size = np.zeros((nt,2))
        step_size[:,0] = step_size_tmp
        step_size[:,1] = step_size_tmp
        
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
        
    ## simulate mvsde
    for i in range(nt):
        xt[i+1,:] = xt[i,:] - alpha*xt[i,:]*dt - beta*(xt[i,:] - x0*np.exp(-alpha*i*dt))*dt + sigma*dwt[i]
        #xt[i+1,:] = xt[i,:] - alpha*xt[i,:]*dt - beta*(xt[i,:] - np.mean(xt[i,:]))*dt + sigma*dwt[i,:]
        
        ## update parameters
        if (est_alpha):
            if norm:
                alpha_est[i+1] = alpha_est[i] + step_size[i,0]*1/N*sigma**(-2)*np.sum(-xt[i,:]*(xt[i+1,:]-xt[i,:])-(-xt[i,:]*(-alpha_est[i]*xt[i,:]-beta_est[i]*(xt[i,:] - x0*np.exp(-alpha_est[i]*i*dt))))*dt)
            else:
                alpha_est[i+1] = alpha_est[i] + step_size[i,0]*sigma**(-2)*(-xt[i,0]*(xt[i+1,0]-xt[i,0])-(-xt[i,0]*(-alpha_est[i]*xt[i,0]-beta_est[i]*(xt[i,0] - x0*np.exp(-alpha_est[i]*i*dt))))*dt)
                                
            if alpha_est[i+1]<0:
                alpha_est[i+1] = alpha_est[i]/2
                
        else:
            alpha_est[i+1] = alpha_est[i]
            
        if (est_beta):
            if norm:
                beta_est[i+1] = beta_est[i] + step_size[i,1]*1/N*sigma**(-2)*np.sum((-(xt[i,:] - x0*np.exp(-alpha_est[i+1]*i*dt)))*(xt[i+1,:]-xt[i,:])-(-(xt[i,:] - x0*np.exp(-alpha_est[i+1]*i*dt)))*(-alpha_est[i+1]*xt[i,:]-beta_est[i]*(xt[i,:] - x0*np.exp(-alpha_est[i+1]*i*dt)))*dt) # sum over all particles
            
            else:
                beta_est[i+1] = beta_est[i] + step_size[i,1]*sigma**(-2)*((-(xt[i,0] - x0*np.exp(-alpha_est[i+1]*i*dt)))*(xt[i+1,0]-xt[i,0])-(-(xt[i,0] - x0*np.exp(-alpha_est[i+1]*i*dt)))*(-alpha_est[i+1]*xt[i,0]-beta_est[i]*(xt[i,0] - x0*np.exp(-alpha_est[i+1]*i*dt)))*dt)
        else:
            beta_est[i+1] = beta_est[i]
                
    
    return beta_est,alpha_est 


##################################



###########################################
#### LOG-LIKELIHOOD (MEAN-FIELD MVSDE) ####
###########################################

def log_lik_mvsde(N,t,xt,dxt,alpha,beta,sigma,x0,dt,marginal = False):
    
    ## compute the log-likelihood at each time
    ll_i = np.zeros(t+1)
    for i in range(t+1):       
        ll_i[i] = -1/(sigma**2)*np.dot(alpha*xt[i,:]+beta*(xt[i,:] - x0*np.exp(-alpha*i*dt)),dxt[i,:])-1/(2*sigma**2)*np.dot(alpha*xt[i,:]+beta*(xt[i,:]-x0*np.exp(-alpha*i*dt)),alpha*xt[i,:]+beta*(xt[i,:]-x0*np.exp(-alpha*i*dt)))*dt
    
    ## sum up log-likelihoods at each time step
    ll = np.sum(ll_i)
    
    ## output
    if marginal:
        return ll, ll_i
    else:
        return ll
    
###########################################
    
    

###################################
#### LOG-LIKELIHOOD (LIJ FORM) ####
###################################

def log_lik_lij(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt,Aij_calc = False, 
                Aij_calc_func = None,Aij_influence_func = None, 
                Aij_scale_param = 1, marginal = False, **kwargs):
    
    ## reshape into matrix if Lij is input as a vector
    if Lij is not None:
        if(Lij.shape == (N*N,)):
            Lij = Lij.reshape(N,N)
    
    ## compute the log-likelihood at each time
    ll_i = np.zeros(t+1)
    for i in range(t+1):
        
        ## compute Aij (and Lij) if necessary
        if(Aij_calc):
            Aij = Aij_calc_func(x=xt[i,:],kernel_func=Aij_influence_func,
                                Aij_scale_param = Aij_scale_param, **kwargs)
            Lij = laplacian_mat(N,Aij)
            
        ll_i[i] = -1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),dxt[i,:])-1/(2*sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]))*dt
    
    ## sum up log-likelihoods at each time step
    ll = np.sum(ll_i)
    
    ## output
    if marginal:
        return ll, ll_i
    else:
        return ll


def log_lik_lij_grad(Lij,N,t,xt,dxt,v_func,alpha,sigma,dt,Aij_calc = False, 
                Aij_calc_func = None,Aij_influence_func = None, 
                Aij_grad_calc_func = None, Aij_grad_influence_func = None,
                n_param = 3,grad_indices = None,Aij_scale_param = 1, **kwargs):
    
    ## reshape into matrix if Lij is input as a vector
    if Lij is not None:
        if(Lij.shape == (N*N,)):
            Lij = Lij.reshape(N,N)
    
    ## dimensions of unknown parameter
    if(Aij_calc):
        dim = n_param
        if grad_indices is None:
            grad_indices = list(range(n_param))
    else:
        dim = N*N
        if grad_indices is None:
            grad_indices = list(range(N*N))
        
    ## initialise gradient of Lij
    Lij_grad = np.zeros((N,N,dim))
    
    ## gradient of Lij (if each entry is unknown parameter)
    if not Aij_calc:
        for i in range(N):
            for j in range(N):
                Lij_grad[i,j,i*N+j] = 1
            
            
    ## gradient at each time step
    ll_grad_i = np.zeros((t+1,dim))

    for i in range(t+1):
        
        ## compute Lij and Lij grad if necessary
        if(Aij_calc):
            Aij = Aij_calc_func(x=xt[i,:],kernel_func=Aij_influence_func,
                                Aij_scale_param = Aij_scale_param,
                                **kwargs)
            Lij = laplacian_mat(N,Aij)
                
            Aij_grad = Aij_grad_calc_func(x=xt[i,:],kernel_func = Aij_influence_func,
                                         kernel_func_grad = Aij_grad_influence_func,
                                         n_param = n_param, grad_indices = grad_indices, 
                                         Aij_scale_param = Aij_scale_param,
                                         **kwargs)
            
            for j in range(dim):
                Lij_grad[:,:,j] = laplacian_mat(N,Aij_grad[:,:,j])
            
        for j in grad_indices:
            ll_grad_i[i,j] = -1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij_grad[:,:,j],xt[i,:]),dxt[i,:])-1/(sigma**2)*np.dot(v_func(xt[i,:],alpha)+np.dot(Lij,xt[i,:]),np.dot(Lij_grad[:,:,j],xt[i,:]))*dt
            
    ## sum up gradients at each time step
    ll_grad = np.sum(ll_grad_i,axis=0)
    
    ## output
    return ll_grad
    
###################################


############################################
#### LOG-LIKELIHOOD (FOR BUMP FUNCTION) ####
############################################

def log_lik_bump(centre, width, squeeze,N,t,xt,dxt,v_func,alpha,sigma,dt,
                 Aij_scale_param = 1):
    
    log_lik = -1/t*log_lik_lij(Lij=None,N=N,t=t,xt=xt,dxt=dxt,v_func=v_func,
                          alpha=alpha,sigma=sigma,dt=dt,Aij_calc = True, 
                          Aij_calc_func = Aij_calc_func,
                          Aij_influence_func = influence_func_bump, 
                          Aij_scale_param = Aij_scale_param, 
                          centre = centre, width = width, squeeze = squeeze)
    
    return log_lik



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
                        Lij_init = None, lasso_t_vals = None,
                        aij_calc = False, aij_func = None,kernel_func = None,
                        n_param = None, param_est_indices = None, 
                        param_init = None, aij_scale_param = 1, 
                        aij_calc_mle_t_vals = None, **kwargs):
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## compute x_t
    xt = sde_sim_func(N=N,T=T,v_func=v_func,alpha=alpha,beta = beta,
                      Lij=Lij,sigma=sigma,x0=x0,dt=dt,seed=seed,
                      Aij_calc=aij_calc, Aij_calc_func = aij_func,
                      Aij_influence_func = kernel_func,
                      Aij_scale_param = aij_scale_param,**kwargs)
    
    ## compute 'dx_t'
    dxt = np.diff(xt,axis=0)
    
    ## compute mle or ridge estimator
    if(mle_ridge):
        
        if aij_calc is False:
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
                
        if aij_calc is True:
            
            n_param_est = len(param_est_indices)
            
            ## costly: only compute at certain times
            estimator = np.zeros((len(aij_calc_mle_t_vals)),n_param_est)
            
            for i in range(len(aij_calc_mle_t_vals)):
                
                nt_i = int(np.round(aij_calc_mle_t_vals[i]/dt))
                xt_i = xt[0:(nt_i+1),:]
                dxt_i = dxt[0:(nt_i),:]
                
                width = list(kwargs.values())[1]
                squeeze = list(kwargs.values())[2]
                
                estimator[i,:] = minimize_scalar(fun = log_lik_bump,
                                                 args = (width, squeeze, N, nt_i-1, 
                                                         xt_i, dxt_i, v_func, alpha, sigma,
                                                         dt, aij_scale_param),
                                                 options = {'maxiter' : 20}).x
            
        
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
             method = "fast", aij_func = None, kernel_func = None,
             aij_grad_func = None, kernel_func_grad = None, n_param = None,
             param_est_init = None, param_est_indices = None, 
             param_true = None, aij_scale_param = 1, **kwargs):
    
    ## set random seed
    np.random.seed(seed)
    
    ## number of time steps
    nt = int(np.round(T/dt))
    
    ## initialise xt
    xt = np.zeros((nt+1,N))
    xt[0,:] = x0
    
    ## intialise beta_est
    Lij_est= np.zeros((nt+1,N,N))
    if Lij_init is not None:
        Lij_est[0,:,:] = Lij_init  
    Lij_est_output = deepcopy(Lij_est)
    
    ## step sizes
    if(type(step_size)==float):
        step_size = [step_size]*nt
    
    ## brownian motion
    dwt = np.sqrt(dt)*np.random.randn(nt+1,N)
    
    
    ## slowest method; assumes that every element of the Lij matrix is an 
    ## independent parameter
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
    
    
    
    ## faster method: assumes that the Lij matrix has a symmetric structure, 
    ## so 'opposite' entries (w.r.t leading diagonal) are equal, and can be 
    ## treated as one parameter; thus Lij only now consists of N(N+1)/2
    ## parameters
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

    
    
    ## fastest method: assumes that the Lij matrix has a symmetric structure
    ## and a certain diagonal structure (i.e. diagonal entires are the 
    ## negative sum of all the other entries on that row); thus Lij only now
    ## consists of N(N-1)/2 parameters
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
       
        
    ## alternative: use this method if we do not simply treat each entry
    ## of the interaction matrix as an unknown parameter, but we assume 
    ## some parametric structure; in this case, Lij_grad may need to be
    ## re-computed at each time step; to do this, we supply the function
    ## lij_grad_func
    if(method=="compute"):
           
        ## initialise parameter estimate
        param_est = np.zeros((nt+1,n_param))
        
        ## initialise other outputs
        aijs = np.zeros((nt+1,N,N))
        lijs = np.zeros((nt+1,N,N))
        aij_grads = np.zeros((nt+1,N,N,n_param))
        lij_grads = np.zeros((nt+1,N,N,n_param))
        
        ## parameter indices (if multiple kernels)
        n_kernels = int(n_param/3)
        centre_indices = list(range(0,n_param,3))
        width_indices = list(range(1,n_param,3))
        squeeze_indices = list(range(2,n_param,3))
        
        for j in range(n_param):
            if j in param_est_indices:
                param_est[0,j] = param_est_init[j]
            
            else:
                param_est[0,j] = param_true[j]
        
        ## iterate
        for i in range(0,nt):
            
            ## compute true Lij (for simulation)
            Aij_true = aij_func(x=xt[i,:],kernel_func=kernel_func,
                                Aij_scale_param = aij_scale_param,
                                **kwargs)
            Lij_true = laplacian_mat(N,Aij_true)
         
            ## current parameters
            centre_tmp = param_est[i,centre_indices]
            width_tmp = param_est[i,width_indices]
            squeeze_tmp = param_est[i,squeeze_indices]
            
            ## compute est Lij (for gradient ascent)
            Aij_est = aij_func(x=xt[i,:],kernel_func=kernel_func,
                               Aij_scale_param = aij_scale_param,
                               centre = centre_tmp, width = width_tmp,
                               squeeze = squeeze_tmp)
            Lij_est = laplacian_mat(N,Aij_est)
            
            aijs[i,:,:] = Aij_est
            lijs[i,:,:] = Lij_est
            
            ## compute Lij_grad
            Aij_grad = aij_grad_func(x=xt[i,:],kernel_func=kernel_func,
                                     kernel_func_grad = kernel_func_grad,
                                     n_param = n_param,
                                     grad_indices = param_est_indices,
                                     Aij_scale_param = aij_scale_param,
                                     centre = centre_tmp,
                                     width = width_tmp,squeeze=squeeze_tmp)
            
            Lij_grad = np.zeros((N,N,n_param))
            for j in range(n_param):
                Lij_grad[:,:,j] = laplacian_mat(N,Aij_grad[:,:,j])
         
            aij_grads[i,:,:,:] = Aij_grad
            lij_grads[i,:,:,:] = Lij_grad
        
            ## simulate SDE
            xt[i+1,:] = xt[i,:] - v_func(xt[i,:],alpha)*dt - np.dot(Lij_true,xt[i,:])*dt + sigma*dwt[i,:]
         
            ## gradient ascent
            for j in range(n_param):
                if j in param_est_indices:
                    param_est[i+1,j] = param_est[i,j] + step_size[i]*sigma**(-2)*(np.dot(-np.dot(Lij_grad[:,:,j],xt[i,:]),(xt[i+1,:]-xt[i,:]))-np.dot((-np.dot(Lij_grad[:,:,j],xt[i,:])),(-v_func(xt[i,:],alpha)-np.dot(Lij_est,xt[i,:])))*dt) 
                    
                else:
                    param_est[i+1,j] = param_est[i,j]
                
        ## output
        Lij_est_output = param_est
            
        
    return Lij_est_output,xt#,aijs,lijs,aij_grads,lij_grads
################################







