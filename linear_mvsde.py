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


def linear_mvsde_online_est_one(xt, theta0, theta_true, est_theta1=True, est_theta2=True, gamma=0.01, sigma=1):

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


def linear_mvsde_online_est_two(xt, theta0, theta_true, est_theta1=True, est_theta2=False, gamma=0.01, sigma=1):

    ## number of time steps
    nt = xt.shape[0] - 1

    ## initialise theta_t, tilde{x}_t, tilde{y}_t
    thetat = np.zeros((nt + 1, 2))
    tildext1 = np.zeros((nt + 1, 1))
    tildext2 = np.zeros((nt + 1, 1))
    tildeyt = np.zeros((nt + 1, 2))

    thetat[0, :] = theta0
    tildext1[0] = xt[0]
    tildext2[0] = xt[0]
    tildeyt[0] = np.zeros(2)

    if est_theta1 is False:
        thetat[0, 0] = theta_true[0]
    if est_theta2 is False:
        thetat[0, 1] = theta_true[1]

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1)

    ## integrate parameter update equations
    for i in tqdm(range(0, nt)):
        t = i * dt
        dxt = xt[i + 1] - xt[i]

        ## tilde x_t1
        tildext1[i + 1] = tildext1[i] - (thetat[i, 0] + thetat[i, 1]) * tildext1[i] * dt + thetat[i, 1] * xt[0] * np.exp(
            -thetat[i, 0] * t) * dt + 1 * sigma * dwt1[i]

        ## tilde x_t2
        tildext2[i + 1] = tildext2[i] - (thetat[i, 0] + thetat[i, 1]) * tildext2[i] * dt + thetat[i, 1] * xt[0] * np.exp(
            -thetat[i, 0] * t) * dt + 1 * sigma * dwt2[i]

        ## tilde y_t
        if est_theta1:
            tildeyt[i + 1, 0] = tildeyt[i, 0] - tildext1[i] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt[i, 0] * dt - \
                                thetat[i, 1] * xt[0] * t * np.exp(-thetat[i, 0] * t) * dt
        if est_theta2:
            tildeyt[i + 1, 1] = tildeyt[i, 1] - tildext1[i] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt[i, 1] * dt + xt[
                0] * np.exp(-thetat[i, 0] * t) * dt

        if est_theta1:
            thetat[i + 1, 0] = thetat[i, 0] + gamma * (-xt[i] + thetat[i, 1] * tildeyt[i, 0]) * (
                        dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext2[i]) * dt)
        else:
            thetat[i + 1, 0] = thetat[i, 0]

        if est_theta2:
            thetat[i + 1, 1] = thetat[i, 1] + gamma * (-xt[i] + tildext1[i] + thetat[i, 1] * tildeyt[i, 1]) * (
                        dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext2[i]) * dt)
        else:
            thetat[i + 1, 1] = thetat[i, 1]
    return thetat


def linear_mvsde_online_est_one_particle_approx(xt, theta0, theta_true, est_theta1, est_theta2, gamma, sigma=1, N=2):

    ## number of time steps
    nt = xt.shape[0] - 1

    ## initialise theta_t, tilde{x}_t^N, tilde{y}_t^N
    thetat = np.zeros((nt + 1, 2))
    tildext_N = np.zeros((nt + 1, 1, N))
    tildeyt_N = np.zeros((nt + 1, 2, N))

    thetat[0, :] = theta0
    tildext_N[0,:] = xt[0] * np.ones(N)
    tildeyt_N[0,:] = np.zeros((2,N))

    if est_theta1 is False:
        thetat[0, 0] = theta_true[0]
    if est_theta2 is False:
        thetat[0, 1] = theta_true[1]

    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    ## integrate parameter update equations
    for i in tqdm(range(0, nt)):
        t = i * dt
        dxt = xt[i + 1] - xt[i]

        ## tilde x_t^N
        tildext_N[i + 1, :] = tildext_N[i, :] - (thetat[i,0] + thetat[i,1]) * tildext_N[i, :] * dt + thetat[i,1] * np.mean(tildext_N[i, :]) * dt + sigma * dwt[i, :]

        ## tilde y_t^N
        if est_theta1:
            tildeyt_N[i + 1, 0, :] = tildeyt_N[i, 0, :] - tildext_N[i, :] * dt - (thetat[i,0] + thetat[i,1]) * tildeyt_N[i, 0, :] * dt + thetat[i, 1] * np.mean(tildeyt_N[i, 0, :]) * dt

        if est_theta2:
            tildeyt_N[i + 1, 1, :] = tildeyt_N[i, 1, :] - tildext_N[i, :] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt_N[i, 1, :] * dt + np.mean(tildext_N[i, :]) * dt + thetat[i, 1] * np.mean(tildeyt_N[i, 1, :]) * dt

        ## theta_t
        if est_theta1:
            thetat[i + 1, 0] = thetat[i, 0] + gamma * (-xt[i] + thetat[i, 1] * np.mean(tildeyt_N[i, 0, :])) * (
                    dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * np.mean(tildext_N[i,:])) * dt)
        else:
            thetat[i + 1, 0] = thetat[i, 0]

        if est_theta2:
            thetat[i + 1, 1] = thetat[i, 1] + gamma * (-xt[i] + np.mean(tildext_N[i, :]) + thetat[i, 1] * np.mean(tildeyt_N[i, 1, :])) * (
                    dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * np.mean(tildext_N[i, :])) * dt)
        else:
            thetat[i + 1, 1] = thetat[i, 1]

    return thetat


def linear_mvsde_online_est_two_particle_approx(xt, theta0, theta_true, est_theta1, est_theta2, gamma, sigma=1, N=2):

    ## number of time steps
    nt = xt.shape[0] - 1

    ## initialise theta_t, tilde{x}_t^N, tilde{y}_t^N
    thetat = np.zeros((nt + 1, 2))
    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildeyt1_N = np.zeros((nt + 1, 2, N))

    thetat[0, :] = theta0
    tildext1_N[0, :] = xt[0] * np.ones(N)
    tildext2_N[0, :] = xt[0] * np.ones(N)
    tildeyt1_N[0,:] = np.zeros((2,N))

    if est_theta1 is False:
        thetat[0, 0] = theta_true[0]
    if est_theta2 is False:
        thetat[0, 1] = theta_true[1]

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    ## integrate parameter update equations
    for i in tqdm(range(0, nt)):
        t = i * dt
        dxt = xt[i + 1] - xt[i]

        ## tilde x_t1^N
        tildext1_N[i + 1, :] = tildext1_N[i, :] - (thetat[i,0] + thetat[i,1]) * tildext1_N[i, :] * dt + thetat[i,1] * np.mean(tildext1_N[i, :]) * dt + sigma * dwt1[i, :]

        ## tilde x_t1^N
        tildext2_N[i + 1, :] = tildext2_N[i, :] - (thetat[i, 0] + thetat[i, 1]) * tildext2_N[i, :] * dt + thetat[i, 1] * np.mean(tildext2_N[i, :]) * dt + sigma * dwt2[i, :]

        ## tilde y_t1^N
        if est_theta1:
            tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - tildext1_N[i, :] * dt - (thetat[i,0] + thetat[i,1]) * tildeyt1_N[i, 0, :] * dt + thetat[i, 1] * np.mean(tildeyt1_N[i, 0, :]) * dt

        if est_theta2:
            tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] - tildext1_N[i, :] * dt - (thetat[i, 0] + thetat[i, 1]) * tildeyt1_N[i, 1, :] * dt + np.mean(tildext1_N[i, :]) * dt + thetat[i, 1] * np.mean(tildeyt1_N[i, 1, :]) * dt

        ## theta_t
        if est_theta1:
            thetat[i + 1, 0] = thetat[i, 0] + gamma * (-xt[i] + thetat[i, 1] * tildeyt1_N[i, 0, 0]) * (
                    dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext2_N[i, 0]) * dt)
        else:
            thetat[i + 1, 0] = thetat[i, 0]

        if est_theta2:
            thetat[i + 1, 1] = thetat[i, 1] + gamma * (-xt[i] + tildext1_N[i, 0] + thetat[i, 1] * tildeyt1_N[i, 1, 0]) * (
                    dxt - (-(thetat[i, 0] + thetat[i, 1]) * xt[i] + thetat[i, 1] * tildext2_N[i, 0]) * dt)
        else:
            thetat[i + 1, 1] = thetat[i, 1]


if __name__ == '__main__':

    # simulation parameters
    N_obs = 1
    T = 10000
    dt = 0.1
    alpha = 0.5
    beta = 0.1
    sigma = 1
    seeds = range(1)

    nt = round(T/dt)
    t = [i * dt for i in range(nt+1)]

    # estimation parameters
    gamma = 0.005
    theta0 = np.array([1.0, 0.8])
    theta_true = np.array([alpha, beta])
    est_theta1 = True
    est_theta2 = False
    N_est = 1

    # plotting
    plot_each_run = False
    plot_mean_run = True

    # output
    save_plots = False
    save_root = "results/linear_mvsde/"

    all_thetat_est1 = np.zeros((nt+1, 2, len(seeds)))
    all_thetat_est2 = np.zeros((nt + 1, 2, len(seeds)))
    all_thetat_est1_approx = np.zeros((nt + 1, 2, len(seeds)))
    all_thetat_est2_approx = np.zeros((nt + 1, 2, len(seeds)))

    for idx, seed in enumerate(seeds):

        print(seed)

        # simulate mvsde
        x0 = np.random.randn(1)
        xt = linear_mvsde_sim_func(N_obs, T, alpha, beta, sigma, x0, dt, seed)

        thetat_est1 = linear_mvsde_online_est_one(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma, sigma)
        thetat_est2 = linear_mvsde_online_est_two(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma, sigma)
        thetat_est1_approx = linear_mvsde_online_est_one_particle_approx(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma, sigma, N_est)
        thetat_est2_approx = linear_mvsde_online_est_two_particle_approx(xt.copy(), theta0, theta_true, est_theta1, est_theta2, gamma, sigma, N_est)

        all_thetat_est1[:,:,idx] = thetat_est1
        all_thetat_est2[:,:,idx] = thetat_est2
        all_thetat_est1_approx[:, :, idx] = thetat_est1_approx
        all_thetat_est2_approx[:, :, idx] = thetat_est2_approx

        if plot_each_run:
            if est_theta1 is True and est_theta2 is False:
                plt.plot(t, thetat_est1[:,0], label=r"$\theta_{t,1}$ (Estimator 1)")
                plt.plot(t, thetat_est2[:,0], label=r"$\theta_{t,1}$ (Estimator 2)")
                plt.plot(t, thetat_est1_approx[:, 0], label=r"$\theta_{t,1}^N$ (Estimator 1, Approx)")
                plt.plot(t, thetat_est2_approx[:, 0], label=r"$\theta_{t,1}^N$ (Estimator 2, Approx)")
                plt.axhline(y=alpha, linestyle="--")
                plt.legend()
                plt.show()
            if est_theta1 is False and est_theta2 is True:
                plt.plot(t, thetat_est1[:,1], label=r"$\theta_{t,2}^N$ (Estimator 1)")
                plt.plot(t, thetat_est2[:,1], label=r"$\theta_{t,2}^N$ (Estimator 2)")
                plt.plot(t, thetat_est1_approx[:, 1], label=r"$\theta_{t,2}^N$ (Estimator 1, Approx)")
                plt.plot(t, thetat_est2_approx[:, 1], label=r"$\theta_{t,2}^N$ (Estimator 2, Approx)")
                plt.axhline(y=beta,linestyle="--")
                plt.legend()
                plt.show()

    if plot_mean_run:
        if est_theta1 is True and est_theta2 is False:
            plt.plot(t, np.mean(all_thetat_est1, 2)[:,0], label=r"$\theta_{t,1}$ (Estimator 1)")
            plt.plot(t, np.mean(all_thetat_est2, 2)[:,0], label=r"$\theta_{t,1}$ (Estimator 2)")
            plt.plot(t, np.mean(all_thetat_est1_approx, 2)[:, 0], label=r"$\theta_{t,1}^N$ (Estimator 1, Approx)")
            plt.plot(t, np.mean(all_thetat_est2_approx, 2)[:, 0], label=r"$\theta_{t,1}^N$ (Estimator 2, Approx)")
            plt.axhline(y=alpha, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(save_root + "alpha_est.eps", dpi=300)
            plt.show()
        elif est_theta1 is False and est_theta2 is True:
            plt.plot(t, np.mean(all_thetat_est1,2)[:,1], label=r"$\theta_{t,2}$ (Estimator 1)")
            plt.plot(t, np.mean(all_thetat_est2,2)[:,1], label=r"$\theta_{t,2}$ (Estimator 2)")
            plt.plot(t, np.mean(all_thetat_est1_approx, 2)[:, 1], label=r"$\theta_{t,2}^N$ (Estimator 1, Approx)")
            plt.plot(t, np.mean(all_thetat_est2_approx, 2)[:, 1], label=r"$\theta_{t,2}^N$ (Estimator 2, Approx)")
            plt.axhline(y=beta, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(save_root + "beta_est.eps", dpi=300)
            plt.show()
        elif est_theta1 is True and est_theta2 is True:
            plt.plot(t, np.mean(all_thetat_est1, 2), label=r"$\theta_{t}$ (Estimator 1)")
            plt.plot(t, np.mean(all_thetat_est2, 2), label=r"$\theta_{t}$ (Estimator 2)")
            plt.plot(t, np.mean(all_thetat_est1_approx, 2), label=r"$\theta_{t}^N$ (Estimator 1, Approx)")
            plt.plot(t, np.mean(all_thetat_est2_approx, 2), label=r"$\theta_{t}^N$ (Estimator 2, Approx)")
            plt.axhline(y=alpha, linestyle="--", color="black")
            plt.axhline(y=beta, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(save_root + "alpha_beta_est.eps", dpi=300)
            plt.show()
