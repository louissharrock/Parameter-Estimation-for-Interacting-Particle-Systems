#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:56:47 2020

@author: ls616
"""

import numpy as np
import matplotlib.pyplot as plt


################################
#### LINEAR MVSDE SIMULATOR ####
################################

## Inputs:
## -> N (number of i.i.d. simulations)
## -> T (length of simulation)
## -> alpha (parameter for confinement potential)
## -> beta (parameter for interaction potential)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)

def linear_mvsde_sim_func(N=1, T=100, alpha=0.5, beta=0.1, sigma=1, x0=1, dt=0.1, seed=1):
    ## set random seed
    np.random.seed(seed)

    ## number of time steps
    nt = int(np.round(T / dt))

    ## initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    ## brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    ## simulate
    for i in range(0, nt):
        t = i * dt
        xt[i+1,:] = xt[i, :] - (alpha + beta) * xt[i, :] * dt + beta * x0 * np.exp(-alpha * t) * dt + sigma * dwt[i, :]

    return xt

#########################


def linear_mvsde_online_est_one(xt, theta0, theta_true, est_theta1=True, est_theta2=True, gamma=0.01):

    ## number of time steps
    nt = xt.shape[0] - 1

    ## initialise theta_t
    thetat = np.zeros((nt+1,2))
    thetat[0,:] = theta0
    if est_theta1 is False:
        thetat[0,0] = theta_true[0]
    if est_theta2 is False:
        thetat[0,1] = theta_true[1]

    ## integrate parameter update equation
    for i in range(0, nt):
        t = i*dt
        dxt = xt[i+1] - xt[i]
        if est_theta1:
            thetat[i + 1, 0] = thetat[i, 0] + gamma * (-xt[i] - thetat[i, 1] * xt[0] * t * np.exp(-thetat[i, 0] * t)) * (
                    dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * xt[0] * np.exp(-thetat[i, 0] * t)) * dt)
        else:
            thetat[i+1, 0] = thetat[i, 0]

        if est_theta2:
            thetat[i + 1, 1] = thetat[i, 1] + gamma * (-xt[i] + xt[0] * np.exp(-thetat[i, 0] * t)) * (dxt - (
                        -(thetat[i+1, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * xt[0] * np.exp(-thetat[i+1, 0] * t)) * dt)
        else:
            thetat[i+1, 1] = thetat[i, 1]
    return thetat


def linear_mvsde_online_est_two(xt, theta0, theta_true, est_theta1=True, est_theta2=False, gamma=0.01):
    



if __name__ == '__main__':

    # set parameters
    N = 1
    T = 10000
    dt = 0.1
    alpha = 0.5
    beta = 0.2
    sigma = 1
    x0 = 2
    seed = 2

    xt = linear_mvsde_sim_func(N, T, alpha, beta, sigma, x0, dt, seed)

    theta0 = np.array([1.0,0.8])
    theta_true = np.array([alpha, beta])
    est_theta1 = True
    est_theta2 = True
    gamma = 0.01

    thetat = linear_mvsde_online_est_one(xt, theta0, theta_true, est_theta1, est_theta2, gamma)
    t = [i*dt for i in range(xt.shape[0])]
    #plt.plot(t, xt)
    plt.plot(t, thetat)


