#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 4 12:56:47 2022

@author: Louis Sharrock
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

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = int(np.round(T / dt))

    # parameters
    if type(alpha) is float:
        alpha = [alpha] * (nt + 1)

    if type(alpha) is int:
        alpha = [alpha] * (nt + 1)

    if type(beta) is float:
        beta = [beta] * (nt + 1)

    if type(beta) is int:
        beta = [beta] * (nt + 1)

    # initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    # brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # simulate
    for i in range(0, nt):
        t = i * dt
        xt[i+1,:] = xt[i, :] - (alpha[i] + beta[i]) * xt[i, :] * dt + beta[i] * x0 * np.exp(-alpha[i] * t) * dt + sigma * dwt[i, :]

    return xt

#########################


################################
#### LINEAR IPS SIMULATOR ####
################################

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> alpha (parameter for confinement potential)
## -> beta (parameter for interaction potential)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)
## Outputs:
## -> xt_1 (a single particle)

def linear_ips_sim_func(N=10, T=100, alpha=0.5, beta=0.1, sigma=1, x0=1, dt=0.1, seed=1):

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = int(np.round(T / dt))

    # parameters
    if type(alpha) is float:
        alpha = [alpha] * (nt + 1)

    if type(alpha) is int:
        alpha = [alpha] * (nt + 1)

    if type(beta) is float:
        beta = [beta] * (nt + 1)

    if type(beta) is int:
        beta = [beta] * (nt + 1)

    # initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    # brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # simulate
    for i in range(0, nt):
        xt[i + 1, :] = xt[i, :] - alpha[i] * xt[i, :] * dt - beta[i] * (xt[i, :] - np.mean(xt[i, :])) * dt + sigma * dwt[i, :]

    # output the first particle
    return xt[:, 0]

#########################


def linear_mvsde_online_est_one(xt, alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma):

    # number of time steps
    nt = xt.shape[0] - 1

    # parameters
    if type(alpha_true) is float:
        alpha_true = [alpha_true] * (nt + 1)

    if type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float:
        beta_true = [beta_true] * (nt + 1)

    if type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    # initialise
    alpha_t = np.zeros(nt + 1)
    if est_alpha:
        alpha_t[0] = alpha0
    else:
        alpha_t = alpha_true

    beta_t = np.zeros(nt + 1)
    if est_beta:
        beta_t[0] = beta0
    else:
        beta_t = beta_true

    # integrate parameter update equation
    for i in tqdm(range(0, nt)):
        t = i*dt
        dxt = xt[i+1] - xt[i]
        if est_alpha:
            alpha_t[i + 1] = alpha_t[i] + gamma * (-xt[i] - beta_t[i] * xt[0] * t * np.exp(-alpha_t[i] * t)) * (
                    dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * xt[0] * np.exp(-alpha_t[i] * t)) * dt)

        if est_beta:
            beta_t[i + 1] = beta_t[i] + gamma * (-xt[i] + xt[0] * np.exp(-alpha_t[i] * t)) * (dxt - (
                        -(alpha_t[i+1] + beta_t[i]) * xt[i] + beta_t[i] * xt[0] * np.exp(-alpha_t[i+1] * t)) * dt)

    return alpha_t, beta_t


def linear_mvsde_online_est_two(xt, alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma, seed=1):

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = xt.shape[0] - 1

    # parameters
    if type(alpha_true) is float:
        alpha_true = [alpha_true] * (nt + 1)

    if type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float:
        beta_true = [beta_true] * (nt + 1)

    if type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    # initialise
    alpha_t = np.zeros(nt + 1)
    if est_alpha:
        alpha_t[0] = alpha0
    else:
        alpha_t = alpha_true

    beta_t = np.zeros(nt + 1)
    if est_beta:
        beta_t[0] = beta0
    else:
        beta_t = beta_true

    tildext1 = np.zeros((nt + 1, 1))
    tildext2 = np.zeros((nt + 1, 1))
    tildext1[0] = xt[0]
    tildext2[0] = xt[0]

    tildeyt = np.zeros((nt + 1, 2))
    tildeyt[0] = np.zeros(2)

    # noise
    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        # tilde x_t1
        tildext1[i + 1] = tildext1[i] - (alpha_t[i] + beta_t[i]) * tildext1[i] * dt + beta_t[i] * xt[0] * np.exp(
            -alpha_t[i] * t) * dt + 1 * sigma * dwt1[i]

        # tilde x_t2
        tildext2[i + 1] = tildext2[i] - (alpha_t[i] + beta_t[i]) * tildext2[i] * dt + beta_t[i] * xt[0] * np.exp(
            -alpha_t[i] * t) * dt + 1 * sigma * dwt2[i]

        # tilde y_t
        if est_alpha:
            tildeyt[i + 1, 0] = tildeyt[i, 0] - tildext1[i] * dt - (alpha_t[i] + beta_t[i]) * tildeyt[i, 0] * dt - \
                                beta_t[i] * xt[0] * t * np.exp(-alpha_t[i] * t) * dt
        if est_beta:
            tildeyt[i + 1, 1] = tildeyt[i, 1] - tildext1[i] * dt - (alpha_t[i] + beta_t[i]) * tildeyt[i, 1] * dt \
                                + xt[0] * np.exp(-alpha_t[i] * t) * dt

        if est_alpha:
            alpha_t[i + 1] = alpha_t[i] + gamma * (-xt[i] + beta_t[i] * tildeyt[i, 0]) * \
                                (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * tildext2[i]) * dt)

        if est_beta:
            beta_t[i + 1] = beta_t[i] + gamma * (-xt[i] + tildext1[i] + beta_t[i] * tildeyt[i, 1]) * \
                               (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * tildext2[i]) * dt)

    return alpha_t, beta_t


def linear_mvsde_online_est_one_particle_approx(xt, alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma,
                                                gamma, N=2, seed=1):
    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = xt.shape[0] - 1

    # parameters
    if type(alpha_true) is float:
        alpha_true = [alpha_true] * (nt + 1)

    if type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float:
        beta_true = [beta_true] * (nt + 1)

    if type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    # initialise
    alpha_t = np.zeros(nt + 1)
    if est_alpha:
        alpha_t[0] = alpha0
    else:
        alpha_t = alpha_true

    beta_t = np.zeros(nt + 1)
    if est_beta:
        beta_t[0] = beta0
    else:
        beta_t = beta_true

    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildext1_N[0, :] = xt[0] * np.ones(N)
    tildext2_N[0, :] = xt[0] * np.ones(N)

    tildeyt1_N = np.zeros((nt + 1, 2, N))
    tildeyt1_N[0, :] = np.zeros((2, N))

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        # tilde x_t1^N
        tildext1_N[i + 1, :] = tildext1_N[i, :] - (alpha_t[i] + beta_t[i]) * tildext1_N[i, :] * dt \
                               + beta_t[i] * np.mean(tildext1_N[i, :]) * dt + sigma * dwt1[i, :]

        # tilde x_t1^N
        tildext2_N[i + 1, :] = tildext2_N[i, :] - (alpha_t[i] + beta_t[i]) * tildext2_N[i, :] * dt \
                               + beta_t[i] * np.mean(tildext2_N[i, :]) * dt + sigma * dwt2[i, :]

        # tilde y_t1^N
        if est_alpha:
            tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - tildext1_N[i, :] * dt \
                                      - (alpha_t[i] + beta_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                      + beta_t[i] * np.mean(tildeyt1_N[i, 0, :]) * dt

        if est_beta:
            tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] - tildext1_N[i, :] * dt \
                                      - (alpha_t[i] + beta_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                      + np.mean(tildext1_N[i, :]) * dt \
                                      + beta_t[i] * np.mean(tildeyt1_N[i, 1, :]) * dt

        # theta_t
        if est_alpha:
            alpha_t[i + 1] = alpha_t[i] + gamma * (-xt[i] + beta_t[i] * np.mean(tildeyt1_N[i, 0, :])) * \
                                (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * np.mean(tildext2_N[i, :])) * dt)

        if est_beta:
            beta_t[i + 1] = beta_t[i] + gamma * (-(xt[i] + np.mean(tildext1_N[i, :])) + beta_t[i] * np.mean(tildeyt1_N[i, 1, :])) \
                               * (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * np.mean(tildext2_N[i, :])) * dt)

    print(tildext1_N)
    print(tildext2_N)
    print(tildeyt1_N)
    print(alpha_t)
    print(beta_t)

    return alpha_t, beta_t


def linear_mvsde_online_est_two_particle_approx(xt, alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma,
                                                gamma, N=2, seed=1):
    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = xt.shape[0] - 1

    # parameters
    if type(alpha_true) is float:
        alpha_true = [alpha_true] * (nt + 1)

    if type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float:
        beta_true = [beta_true] * (nt + 1)

    if type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    # initialise
    alpha_t = np.zeros(nt + 1)
    if est_alpha:
        alpha_t[0] = alpha0
    else:
        alpha_t = alpha_true

    beta_t = np.zeros(nt + 1)
    if est_beta:
        beta_t[0] = beta0
    else:
        beta_t = beta_true

    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildext1_N[0, :] = xt[0] * np.ones(N)
    tildext2_N[0, :] = xt[0] * np.ones(N)

    tildeyt1_N = np.zeros((nt + 1, 2, N))
    tildeyt1_N[0, :] = np.zeros((2, N))

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        # tilde x_t1^N
        tildext1_N[i + 1, :] = tildext1_N[i, :] - (alpha_t[i] + beta_t[i]) * tildext1_N[i, :] * dt \
                               + beta_t[i] * np.mean(tildext1_N[i, :]) * dt + sigma * dwt1[i, :]

        # tilde x_t1^N
        tildext2_N[i + 1, :] = tildext2_N[i, :] - (alpha_t[i] + beta_t[i]) * tildext2_N[i, :] * dt \
                               + beta_t[i] * np.mean(tildext2_N[i, :]) * dt + sigma * dwt2[i, :]

        # tilde y_t1^N
        if est_alpha:
            tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - tildext1_N[i, :] * dt \
                                      - (alpha_t[i] + beta_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                      + beta_t[i] * np.mean(tildeyt1_N[i, 0, :]) * dt

        if est_beta:
            tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] - tildext1_N[i, :] * dt \
                                      - (alpha_t[i] + beta_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                      + np.mean(tildext1_N[i, :]) * dt \
                                      + beta_t[i] * np.mean(tildeyt1_N[i, 1, :]) * dt

        # alpha_t
        if est_alpha:
            alpha_t[i + 1] = alpha_t[i] + gamma * (-xt[i] + beta_t[i] * tildeyt1_N[i, 0, 0]) * \
                             (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * tildext2_N[i, 0]) * dt)

        # beta_t
        if est_beta:
            beta_t[i + 1] = beta_t[i] + gamma * (-xt[i] + tildext1_N[i, 0] + beta_t[i] * tildeyt1_N[i, 1, 0]) * \
                            (dxt - (-(alpha_t[i] + beta_t[i]) * xt[i] + beta_t[i] * tildext2_N[i, 0]) * dt)

    return alpha_t, beta_t


if __name__ == '__main__':

    # simulation parameters
    N_obs = 1
    N_par = 100
    T = 20000
    dt = 0.1
    alpha = 1.5
    beta = 0.7
    sigma = 1
    seeds = range(10)

    nt = round(T/dt)
    t = [i * dt for i in range(nt+1)]

    # estimation parameters
    gamma = 0.005

    alpha0 = 2.0
    alpha_true = alpha
    est_alpha = True

    beta0 = 0.2
    beta_true = beta
    est_beta = False

    N_est = 2

    # plotting
    plot_each_run = False
    plot_mean_run = True

    # observations
    observations = ['linear_mvsde', 'linear_ips']

    # output
    save_plots = True

    for obs in observations:

        save_root = "results/" + obs + "/"

        all_alpha_est1 = np.zeros((nt + 1, len(seeds)))
        all_beta_est1 = np.zeros((nt + 1, len(seeds)))

        all_alpha_est2 = np.zeros((nt + 1, len(seeds)))
        all_beta_est2 = np.zeros((nt + 1, len(seeds)))

        all_alpha_est1_approx = np.zeros((nt + 1, len(seeds)))
        all_beta_est1_approx = np.zeros((nt + 1, len(seeds)))

        all_alpha_est2_approx = np.zeros((nt + 1, len(seeds)))
        all_beta_est2_approx = np.zeros((nt + 1, len(seeds)))

        for idx, seed in enumerate(seeds):

            print(seed)

            # simulate mvsde
            x0 = np.random.randn(1)
            if obs == "linear_mvsde":
                xt = linear_mvsde_sim_func(N_obs, T, alpha, beta, sigma, x0, dt, seed)
            if obs == "linear_ips":
                xt = linear_ips_sim_func(N_par, T, alpha, beta, sigma, x0, dt, seed)

            alpha_est1, beta_est1 = linear_mvsde_online_est_one(xt.copy(), alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma)
            alpha_est2, beta_est2 = linear_mvsde_online_est_two(xt.copy(), alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma)
            alpha_est1_approx, beta_est1_approx = linear_mvsde_online_est_one_particle_approx(xt.copy(), alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma, N_est)
            alpha_est2_approx, beta_est2_approx = linear_mvsde_online_est_two_particle_approx(xt.copy(), alpha0, alpha_true, est_alpha, beta0, beta_true, est_beta, sigma, gamma, N_est)

            all_alpha_est1[:, idx], all_beta_est1[:, idx] = alpha_est1, beta_est1
            all_alpha_est2[:, idx], all_beta_est2[:, idx] = alpha_est2, beta_est2
            all_alpha_est1_approx[:, idx], all_beta_est1_approx[:, idx] = alpha_est1_approx, beta_est1_approx
            all_alpha_est2_approx[:, idx], all_beta_est2_approx[:, idx] = alpha_est2_approx, beta_est2_approx

            if plot_each_run:
                if est_alpha and not est_beta:
                    plt.plot(t, alpha_est1, label=r"$\theta_{t,1}$ (Estimator 1)")
                    plt.plot(t, alpha_est2, label=r"$\theta_{t,1}$ (Estimator 2)")
                    plt.plot(t, alpha_est1_approx, label=r"$\theta_{t,1}^N$ (Estimator 1, Approx)")
                    plt.plot(t, alpha_est2_approx, label=r"$\theta_{t,1}^N$ (Estimator 2, Approx)")
                    plt.axhline(y=alpha, linestyle="--")
                    plt.legend()
                    plt.show()
                if est_beta and not est_alpha:
                    plt.plot(t, beta_est1, label=r"$\theta_{t,2}^N$ (Estimator 1)")
                    plt.plot(t, beta_est2, label=r"$\theta_{t,2}^N$ (Estimator 2)")
                    plt.plot(t, beta_est1_approx, label=r"$\theta_{t,2}^N$ (Estimator 1, Approx)")
                    plt.plot(t, beta_est2_approx, label=r"$\theta_{t,2}^N$ (Estimator 2, Approx)")
                    plt.axhline(y=beta,linestyle="--")
                    plt.legend()
                    plt.show()

        if plot_mean_run:
            if est_alpha and not est_beta:
                plt.plot(t, np.mean(all_alpha_est1, 1), label=r"$\theta_{t,1}$ (Estimator 1)")
                plt.plot(t, np.mean(all_alpha_est2, 1), label=r"$\theta_{t,1}$ (Estimator 2)")
                plt.plot(t, np.mean(all_alpha_est1_approx, 1), label=r"$\theta_{t,1}^N$ (Estimator 1, Approx)")
                plt.plot(t, np.mean(all_alpha_est2_approx, 1), label=r"$\theta_{t,1}^N$ (Estimator 2, Approx)")
                plt.axhline(y=alpha, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(save_root + "alpha_est_all.eps", dpi=300)
                plt.show()
            elif est_beta and not est_alpha:
                plt.plot(t, np.mean(all_beta_est1, 1), label=r"$\theta_{t,2}$ (Estimator 1)")
                plt.plot(t, np.mean(all_beta_est2, 1), label=r"$\theta_{t,2}$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_est1_approx, 1), label=r"$\theta_{t,2}^N$ (Estimator 1, Approx)")
                plt.plot(t, np.mean(all_beta_est2_approx, 1), label=r"$\theta_{t,2}^N$ (Estimator 2, Approx)")
                plt.axhline(y=beta, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(save_root + "beta_est_all.eps", dpi=300)
                plt.show()