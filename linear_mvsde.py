#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:56:47 2020

@author: ls616
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    for i in tqdm(range(0, nt)):
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

    ## number of time steps
    nt = xt.shape[0] - 1

    ## initialise theta_t, tilde{x}_t, tilde{y}_t
    thetat = np.zeros((nt + 1, 2))
    tildext = np.zeros((nt + 1, 1))
    tildeyt = np.zeros((nt + 1, 2))

    thetat[0, :] = theta0
    tildext[0] = xt[0]
    tildeyt[0] = np.zeros(2)

    if est_theta1 is False:
        thetat[0, 0] = theta_true[0]
    if est_theta2 is False:
        thetat[0, 1] = theta_true[1]

    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    ## integrate parameter update equations
    for i in tqdm(range(0, nt)):
        t = i * dt
        dxt = xt[i + 1] - xt[i]

        ## tilde x_t
        tildext[i + 1] = tildext[i] - (thetat[i, 0] + thetat[i, 1]) * tildext[i] * dt + thetat[i, 1] * xt[0] * np.exp(
            -thetat[i, 0] * t) * dt + 0.2 * sigma * dwt[i,:]

        ## tilde y_t
        if est_theta1:
            tildeyt[i + 1, 0] = tildeyt[i, 0] - tildext[i] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt[i, 0] * dt - \
                                thetat[i, 1] * xt[0] * t * np.exp(-thetat[i, 0] * t) * dt
        if est_theta2:
            tildeyt[i + 1, 1] = tildeyt[i, 1] - tildext[i] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt[i, 1] * dt + xt[
                0] * np.exp(-thetat[i, 0] * t) * dt

        if est_theta1:
            thetat[i + 1, 0] = thetat[i, 0] + gamma * (-xt[i] + thetat[i, 1] * tildeyt[i, 0]) * (
                        dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext[i]) * dt)
        else:
            thetat[i + 1, 0] = thetat[i, 0]

        if est_theta2:
            thetat[i + 1, 1] = thetat[i, 1] + gamma * (-xt[i] + tildext[i] + thetat[i, 1] * tildeyt[i, 1]) * (
                        dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext[i]) * dt)
        else:
            thetat[i + 1, 1] = thetat[i, 1]
    return thetat





if __name__ == '__main__':

    # simulation parameters
    N = 1
    T = 10000
    dt = 0.1
    alpha = 1
    beta = 0.2
    sigma = 1
    x0 = 1
    seeds = range(1)

    nt = round(T/dt)
    t = [i * dt for i in range(nt+1)]

    # estimation parameters
    gamma = 0.005
    theta0 = np.array([1.0, 0.5])
    theta_true = np.array([alpha, beta])
    est_theta1 = False
    est_theta2 = True

    # plotting
    plot_each_run = False
    plot_mean_run = True

    all_thetat_est1 = np.zeros((nt+1,2,len(seeds)))
    all_thetat_est2 = np.zeros((nt + 1, 2, len(seeds)))

    for idx, seed in enumerate(seeds):

        print(seed)

        # simulate mvsde
        xt = linear_mvsde_sim_func(N, T, alpha, beta, sigma, x0, dt, seed)

        thetat_est1 = linear_mvsde_online_est_one(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma)
        thetat_est2 = linear_mvsde_online_est_two(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma)

        all_thetat_est1[:,:,idx] = thetat_est1
        all_thetat_est2[:,:,idx] = thetat_est2

        if plot_each_run:
            if est_theta1 is True and est_theta2 is False:
                plt.plot(t, thetat_est1[:,0], label="Est1")
                plt.plot(t, thetat_est2[:,0], label="Est2")
                plt.axhline(y=alpha, linestyle="--")
                plt.legend()
                plt.show()
            if est_theta1 is False and est_theta2 is True:
                plt.plot(t, thetat_est1[:,1], label="Est1")
                plt.plot(t, thetat_est2[:,1], label="Est2")
                plt.axhline(y=beta,linestyle="--")
                plt.legend()
                plt.show()

    if plot_mean_run:
        if est_theta1 is True and est_theta2 is False:
            plt.plot(t, np.mean(all_thetat_est1,2)[:,0], label="Est1")
            plt.plot(t, np.mean(all_thetat_est2,2)[:,0], label="Est2")
            plt.axhline(y=alpha, linestyle="--")
            plt.legend()
            plt.show()
        elif est_theta1 is False and est_theta2 is True:
            plt.plot(t, np.mean(all_thetat_est1,2)[:,1], label="Est1")
            plt.plot(t, np.mean(all_thetat_est2,2)[:,1], label="Est2")
            plt.axhline(y=beta, linestyle="--")
            plt.legend()
            plt.show()
        elif est_theta1 is True and est_theta2 is True:
            plt.plot(t, np.mean(all_thetat_est1, 2), label="Est1")
            plt.plot(t, np.mean(all_thetat_est2, 2), label="Est2")
            plt.axhline(y=alpha, linestyle="--", color="black")
            plt.axhline(y=beta, linestyle="--", color="black")
            plt.legend()
            plt.show()




