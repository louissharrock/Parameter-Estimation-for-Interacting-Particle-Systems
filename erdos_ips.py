#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 4 12:56:47 2022

@author: Louis Sharrock
"""

import numpy as np
import matplotlib.pyplot as plt

## erdos-renyi


def random_adjacency_matrix(N, p):
    matrix = np.array([[np.random.binomial(size=1, n=1, p=p)[0] for i in range(N)] for j in range(N)])

    # No vertex connects to itself
    for i in range(N):
        matrix[i][i] = 1

    # If i is connected to j, j is connected to i
    for i in range(N):
        for j in range(N):
            matrix[j][i] = matrix[i][j]

    return matrix

###################################
#### ERDOS-RENYI IPS SIMULATOR ####
###################################

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> alpha (parameter for confinement potential)
## -> beta (parameter for interaction potential)
## -> p (paramater for erdos renyi graph)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)

def erdos_renyi_ips_sim_func(N, T, alpha, beta, p, sigma=1, x0=1, dt=0.1, seed=1):

    ## set random seed
    np.random.seed(seed)

    ## number of time steps
    nt = int(np.round(T / dt))

    ## initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    ## brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    ## generate erdos renyi graph
    Aij = random_adjacency_matrix(N, p)
    Ni = np.sum(Aij,axis=0)
    print(Aij[0,:])
    print("p =",p)
    print("Ni/N =",np.round(np.mean((Ni-1)/(N-1)),2))


    ## simulate
    for i in range(0, nt):

        for j in range(0, N):

            xt[i + 1, j] = xt[i, j] - alpha * xt[i,j] * dt - beta / Ni[j] * np.dot(Aij[j, :], xt[i, j] - xt[i, :]) * dt + sigma * dwt[i, j]

    return xt

#######################


if __name__ == '__main__':

    # parameters
    N = 30
    T = 5
    alpha = 0
    beta = 1.0
    p_vals = [0,0.02,0.05,0.1]
    sigma = 1
    x0 = np.random.normal(0,20,N)
    dt = 0.1
    seed = 0

    nt = round(T / dt)
    t = [i * dt for i in range(nt + 1)]

    xt_all = []

    for p in p_vals:
        xt = erdos_renyi_ips_sim_func(N, T, alpha, beta, p, sigma, x0, dt, seed)
        xt_all.append(xt)

    fig, axs = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            axs[i,j].plot(t,xt_all[2*i+j])
            axs[i,j].set_xlabel("t")