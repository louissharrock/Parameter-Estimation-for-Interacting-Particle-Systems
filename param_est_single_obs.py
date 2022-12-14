import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


#############################
#### POTENTIAL FUNCTIONS ####
#############################

## Gradient of quadratic potential
def grad_quadratic(x, alpha):
    return alpha * x

def grad_theta_grad_quadratic(x, alpha):
    return x

def grad_x_grad_quadratic(x, alpha):
    return alpha


## Gradient of linear potential (null)
def grad_linear(x, alpha):
    return 0

def grad_theta_grad_linear(x, alpha):
    return 0
def grad_x_grad_linear(x, alpha):
    return 0


## Gradient of bi-stable (Landau) potential
def grad_bi_stable(x, alpha):
    return alpha * (x ** 3 - x)

def grad_theta_grad_bi_stable(x, alpha):
    return x**3 - x

def grad_x_grad_bi_stable(x, alpha):
    return alpha * (3 * x ** 2 - 1)


## Gradient of sine (Kuramoto) potential
def grad_kuramoto(x, alpha):
    return alpha * np.sin(x)

def grad_theta_grad_kuramoto(x, alpha):
    return np.sin(x)

def grad_x_grad_kuramoto(x, alpha):
    return alpha * np.cos(x)

#############################

#######################
#### IPS SIMULATOR ####
#######################

## Simulate the IPS with N particles, and output a single observation

## We use this as an approximation to the MVSDE, given that in most cases we cannot simulate
## the MVSDE directly (other than the linear case; see linear_mvsde.py)

## Inputs:
## -> N (number of particles)
## -> T (length of simulation)
## -> grad_v (gradient of confinement potential)
## -> alpha (parameter for confinement potential)
## -> grad_w (gradient of interaction potential, optional)
## -> beta (paramater for interaction potential, optional)
## -> Aij (interaction matrix, optional)
## -> Lij (laplacian matrix), optional)
## -> sigma (noise magnitude)
## -> x0  (initial value)
## -> dt (time step)
## -> seed (random seed)

## Outputs:
## -> x_t^{i} = x_t^{1} (a single particle from the IPS)

def sde_sim_func(N=20, T=100, grad_v=grad_quadratic, alpha=1,
                 grad_w=grad_quadratic, beta=0.1, Aij=None,
                 Lij=None, sigma=1, x0=1, dt=0.1, seed=1,
                 Aij_calc=False, Aij_calc_func=None,
                 Aij_influence_func=None, Aij_scale_param=1,
                 kuramoto=False, fitzhugh=False, y0=1, gamma=2,
                 **kwargs):

    # check inputs
    if fitzhugh:
        assert y0 is not None
        assert gamma is not None

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

    if fitzhugh:
        if type(gamma) is int:
            gamma = [gamma] * (nt+1)
        if type(gamma) is float:
            gamma = [gamma] * (nt+1)

    # initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    # intialise yt
    if fitzhugh:
        yt = np.zeros((nt+1, N))
        yt[0, :] = y0

    # brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # simulate

    # if interaction parameter provided
    if beta is not None:
        if grad_w == grad_quadratic:
            if not fitzhugh:
                for i in tqdm(range(0, nt)):
                    xt[i + 1, :] = xt[i, :] - grad_v(xt[i, :], alpha[i]) * dt - beta[i] * (
                                xt[i, :] - np.mean(xt[i, :])) * dt + sigma * dwt[i, :]
            if fitzhugh:
                for i in tqdm(range(0, nt)):
                    xt[i + 1, :] = xt[i, :] - alpha[i] * ((1 / 3 * xt[i, :] ** 3 - xt[i, :]) + yt[i, :]) * dt \
                                   - beta[i] * (xt[i, :] - np.mean(xt[i, :])) * dt + sigma * dwt[i, :]
                    yt[i + 1, :] = yt[i, :] + (gamma[i] + xt[i, :]) * dt

        else:
            for i in tqdm(range(0, nt)):
                for j in range(N):
                    xt[i + 1, j] = xt[i, j] - grad_v(xt[i, j], alpha[i]) * dt - 1 / N * np.sum(
                        grad_w(xt[i, j] - xt[i, :], beta[i])) * dt + sigma * dwt[i, j]
                if kuramoto:
                    while np.any(xt[i + 1, :] > + np.pi) or np.any(xt[i + 1, :] < - np.pi):
                        xt[i + 1, np.where(xt[i + 1, :] > +np.pi)] -= 2. * np.pi
                        xt[i + 1, np.where(xt[i + 1, :] < -np.pi)] += 2. * np.pi

    # if in Aij form
    if np.any(Aij) or Aij_calc:
        for i in tqdm(range(0, nt)):

            # compute Aij if necessary
            if (Aij_calc):
                Aij = Aij_calc_func(x=xt[i, :], kernel_func=Aij_influence_func,
                                    Aij_scale_param=Aij_scale_param, **kwargs)

            # simulate
            for j in range(0, N):
                xt[i + 1, j] = xt[i, j] - grad_v(xt[i, j], alpha[i]) * dt - \
                               np.dot(Aij[j, :],xt[i, j] - xt[i, :]) * dt + sigma * dwt[i, j]

    # if in Lij form
    if np.any(Lij):
        for i in tqdm(range(0, nt)):
            xt[i + 1, :] = xt[i, :] - grad_v(xt[i, :], alpha[i]) * dt - np.dot(Lij, xt[i, :]) * dt + sigma * dwt[i, :]

    if fitzhugh:
        return xt[:, 0], yt[:, 0]
    else:
        return xt[:, 0]

#######################


############################
#### ONLINE ESTIMATOR 1 ####
############################

## Recursive MLE (1)

## Inputs:
## -> xt (observed path of MVSDE)
## -> grad_v (gradient of confinement potential)
## -> grad_theta_grad_v (gradient of gradient of confinement potential w.r.t theta)
## -> alpha0 (initial parameter for confinement potential)
## -> alpha_true (true parameter for confinement potential)
## -> est_alpha (whether to estimate parameter for confinement potential)
## -> grad_w (gradient of interaction potential, optional)
## -> grad_theta_grad_w (gradient of gradient of interaction potential w.r.t theta)
## -> beta0 (initial paramater for interaction potential)
## -> beta_true (true parameter for interaction potential)
## -> est_beta (whether to estimate parameter for interaction potential)
## -> sigma (noise magnitude)
## -> gamma (learning rate)
## -> N (numnber of synthetic particles to use in the estimator)

## Outputs:
## -> theta_t (online parameter estimate)

def online_est_one(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true, est_alpha, grad_w,
                   grad_theta_grad_w, grad_x_grad_w, beta0, beta_true, est_beta, sigma, gamma, N=2, seed=1,
                   fitzhugh=False, yt=None, gamma0=None, gamma_true=None, est_gamma=False, kuramoto=False):

    # check inputs
    if fitzhugh:
        assert yt is not None
        assert gamma0 is not None
        assert gamma_true is not None

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

    if fitzhugh:
        if type(gamma_true) is float:
            gamma_true = [gamma_true] * (nt + 1)

        if type(gamma_true) is int:
            gamma_true = [gamma_true] * (nt + 1)

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

    if fitzhugh:
        gamma_t = np.zeros(nt + 1)
        if est_gamma:
            gamma_t[0] = gamma0
        else:
            gamma_t = gamma_true

    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildext1_N[0, :] = xt[0] * np.ones(N)
    tildext2_N[0, :] = xt[0] * np.ones(N)

    tildeyt1_N = np.zeros((nt + 1, 2, N))
    tildeyt1_N[0, :] = np.zeros((2, N))

    if fitzhugh:
        tildext1_N_y = np.zeros((nt + 1, N))
        tildext2_N_y = np.zeros((nt + 1, N))
        tildext1_N_y[0, :] = yt[0] * np.ones(N)
        tildext2_N_y[0, :] = yt[0] * np.ones(N)

        tildeyt1_N = np.zeros((nt + 1, 3, N)) # tangent of x process; now have 3 parameters
        tildeyt1_N[0, :] = np.zeros((3, N))

        tildeyt1_N_y = np.zeros((nt + 1, 3, N)) # tangent of y process; now have 3 parameters
        tildeyt1_N_y[0, :] = np.zeros((3, N))

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        if fitzhugh:
            dyt = yt[i+1] - yt[i]

        # tilde x_t1^N and tilde x_t2^N
        if grad_w == grad_quadratic:
            if not fitzhugh:
                tildext1_N[i + 1, :] = tildext1_N[i, :] - grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                       + sigma * dwt1[i, :]
                tildext2_N[i + 1, :] = tildext2_N[i, :] - grad_v(tildext2_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                       + sigma * dwt2[i, :]
            if fitzhugh:
                tildext1_N[i + 1, :] = tildext1_N[i, :] - alpha_t[i] * (1 / 3 * tildext1_N[i, :] ** 3 - tildext1_N[i, :]
                                                                        + tildext1_N_y[i, :]) * dt \
                                       - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                       + sigma * dwt1[i, :]
                tildext1_N_y[i + 1, :] = tildext1_N_y[i, :] + (gamma_t[i] + tildext1_N[i, :]) * dt

                tildext2_N[i + 1, :] = tildext2_N[i, :] - alpha_t[i] * (1 / 3 * tildext2_N[i, :] ** 3 - tildext2_N[i, :]
                                                                        + tildext2_N_y[i, :]) * dt \
                                       - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                       + sigma * dwt2[i, :]
                tildext2_N_y[i + 1, :] = tildext2_N_y[i, :] + (gamma_t[i] + tildext2_N[i, :]) * dt

        else:
            for j in range(N):
                tildext1_N[i + 1, j] = tildext1_N[i, j] - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt1[i, j]
                tildext2_N[i + 1, j] = tildext2_N[i, j] - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt2[i, j]

            if kuramoto:
                while np.any(tildext1_N[i + 1, :] > + np.pi) or np.any(tildext1_N[i + 1, :] < - np.pi):
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] < -np.pi)] += 2. * np.pi
                while np.any(tildext2_N[i + 1, :] > + np.pi) or np.any(tildext2_N[i + 1, :] < - np.pi):
                    tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                    tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] < -np.pi)] += 2. * np.pi

        # tilde y_t1^N
        if est_alpha:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - grad_theta_grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
                if fitzhugh:
                    tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] \
                                              - (1 / 3 * tildext1_N[i, :] ** 3 - tildext1_N[i, :] + tildext1_N_y[i, :]) * dt \
                                              - alpha_t[i] * ((tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 0, :] + tildeyt1_N_y[i, 0, :]) * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
                    tildeyt1_N_y[i + 1, 0, :] = tildeyt1_N_y[i, 0, :] + tildeyt1_N[i+1, 0, :] * dt

            else:
                for j in range(N):
                    tildeyt1_N[i + 1, 0, j] = tildeyt1_N[i, 0, j] \
                                              - grad_theta_grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 0, j] * dt \
                                              - 1 / N * np.sum(tildeyt1_N[i, 0, j] * grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[1, 0, :] * - grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt

        if est_beta:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                              - (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt
                if fitzhugh:
                    tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                              - alpha_t[i] * ((tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 1, :] + tildeyt1_N_y[i, 1, :]) * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                              - (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt
                    tildeyt1_N_y[i + 1, 1, :] = tildeyt1_N_y[i, 1, :] + tildeyt1_N[i+1, 1, :] * dt

            else:
                for j in range(N):
                    tildeyt1_N[i + 1, 1, j] = tildeyt1_N[i, 1, j] \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 1, j] * dt \
                                              - 1 / N * np.sum(grad_theta_grad_w(tildext1_N[i,j] - tildext1_N[i,:], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[i, 1, j] * grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[1, 1, :] * - grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt

        if est_gamma:
            tildeyt1_N[i + 1, 2, :] = tildeyt1_N[i, 2, :] \
                                      - alpha_t[i] * ((tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 2, :] + tildeyt1_N_y[i, 2, :]) * dt \
                                      - beta_t[i] * (tildeyt1_N[i, 2, :] - np.mean(tildeyt1_N[i, 2, :])) * dt
            tildeyt1_N_y[i + 1, 2, :] = tildeyt1_N_y[i, 2, :] + (1 + tildeyt1_N[i+1, 2, :]) * dt

        # alpha_t
        if est_alpha:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    alpha_t[i+1] = alpha_t[i] + gamma \
                                   * (- grad_theta_grad_v(xt[i], alpha_t[i]) + beta_t[i] * np.mean(tildeyt1_N[i, 0, :])) \
                                   * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)
                if fitzhugh:
                    alpha_t[i + 1] = alpha_t[i] + gamma \
                                     * (-(1 / 3 * xt[i] ** 3 - xt[i] + yt[i]) + beta_t[i] * np.mean(tildeyt1_N[i, 0, :])) \
                                     * (dxt - (- alpha_t[i] * (1 / 3 * xt[i] ** 3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)

            else:
                alpha_t[i+1] = alpha_t[i] + gamma \
                               * (- grad_theta_grad_v(xt[i], alpha_t[i])
                                  + 1 / N * np.sum(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                               * 1/(sigma**2) * (dxt - (-grad_v(xt[i], alpha_t[i]) - 1 / N * np.sum(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)

        # beta_t
        if est_beta:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    beta_t[i + 1] = beta_t[i] + gamma \
                                    * (- (xt[i] - np.mean(tildext1_N[i, :])) + beta_t[i] * np.mean(tildeyt1_N[i, 1, :])) \
                                    * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)
                if fitzhugh:
                    beta_t[i + 1] = beta_t[i] + gamma \
                                    * (- (xt[i] - tildext1_N[i, 0]) + beta_t[i] * np.mean(tildeyt1_N[i, 1, :])) \
                                    * (dxt - (- alpha_t[i] * (1 / 3 * xt[i] ** 3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)

            else:
                beta_t[i + 1] = beta_t[i] + gamma \
                               * (- 1 / N * np.sum(grad_theta_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))
                                  + 1 / N * np.sum(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                               * 1/(sigma**2) * (dxt - (-grad_v(xt[i], alpha_t[i]) - 1 / N * np.sum(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)

        if est_gamma:
            gamma_t[i + 1] = gamma_t[i] \
                             + gamma \
                             * (beta_t[i] * tildeyt1_N[i, 2, 0]) \
                             * (dxt - (- alpha_t[i] * (1 / 3 * xt[i] ** 3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt) \
                             + gamma \
                             * (dyt - (gamma_t[i] + xt[i]) * dt)

    if not fitzhugh:
        return alpha_t, beta_t
    if fitzhugh:
        return alpha_t, beta_t, gamma_t

#######################


def online_est_two(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true, est_alpha, grad_w,
                   grad_theta_grad_w, grad_x_grad_w, beta0, beta_true, est_beta, sigma, gamma, N=2, seed=1,
                   fitzhugh=False, yt=None, gamma0=None, gamma_true=None, est_gamma=False, kuramoto=False):
    # check inputs
    if fitzhugh:
        assert yt is not None
        assert gamma0 is not None
        assert gamma_true is not None

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

    if fitzhugh:
        if type(gamma_true) is float:
            gamma_true = [gamma_true] * (nt + 1)

        if type(gamma_true) is int:
            gamma_true = [gamma_true] * (nt + 1)

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

    if fitzhugh:
        gamma_t = np.zeros(nt + 1)
        if est_gamma:
            gamma_t[0] = gamma0
        else:
            gamma_t = gamma_true

    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildext1_N[0, :] = xt[0] * np.ones(N)
    tildext2_N[0, :] = xt[0] * np.ones(N)

    tildeyt1_N = np.zeros((nt + 1, 2, N))
    tildeyt1_N[0, :] = np.zeros((2, N))

    if fitzhugh:
        tildext1_N_y = np.zeros((nt + 1, N))
        tildext2_N_y = np.zeros((nt + 1, N))
        tildext1_N_y[0, :] = yt[0] * np.ones(N)
        tildext2_N_y[0, :] = yt[0] * np.ones(N)

        tildeyt1_N = np.zeros((nt + 1, 3, N))  # tangent of x process; now have 3 parameters
        tildeyt1_N[0, :] = np.zeros((3, N))

        tildeyt1_N_y = np.zeros((nt + 1, 3, N))  # tangent of y process; now have 3 parameters
        tildeyt1_N_y[0, :] = np.zeros((3, N))

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        if fitzhugh:
            dyt = yt[i+1] - yt[i]

        # tilde x_t1^N and tilde x_t2^N
        if grad_w == grad_quadratic:
            if not fitzhugh:
                tildext1_N[i + 1, :] = tildext1_N[i, :] - grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                       + sigma * dwt1[i, :]
                tildext2_N[i + 1, :] = tildext2_N[i, :] - grad_v(tildext2_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                       + sigma * dwt2[i, :]
            if fitzhugh:
                tildext1_N[i + 1, :] = tildext1_N[i, :] - alpha_t[i] * (1 / 3 * tildext1_N[i, :] ** 3 - tildext1_N[i, :]
                                                                        + tildext1_N_y[i, :]) * dt \
                                       - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                       + sigma * dwt1[i, :]
                tildext1_N_y[i + 1, :] = tildext1_N_y[i, :] + (gamma_t[i] + tildext1_N[i, :]) * dt

                tildext2_N[i + 1, :] = tildext2_N[i, :] - alpha_t[i] * (1 / 3 * tildext2_N[i, :] ** 3 - tildext2_N[i, :]
                                                                        + tildext2_N_y[i, :]) * dt \
                                       - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                       + sigma * dwt2[i, :]
                tildext2_N_y[i + 1, :] = tildext2_N_y[i, :] + (gamma_t[i] + tildext2_N[i, :]) * dt

        else:
            for j in range(N):
                tildext1_N[i + 1, j] = tildext1_N[i, j] - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt1[i, j]
                tildext2_N[i + 1, j] = tildext2_N[i, j] - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt2[i, j]
            if kuramoto:
                while np.any(tildext1_N[i + 1, :] > + np.pi) or np.any(tildext1_N[i + 1, :] < - np.pi):
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] < -np.pi)] += 2. * np.pi
                while np.any(tildext2_N[i + 1, :] > + np.pi) or np.any(tildext2_N[i + 1, :] < - np.pi):
                    tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                    tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] < -np.pi)] += 2. * np.pi

        # tilde y_t1^N
        if est_alpha:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - grad_theta_grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
                if fitzhugh:
                    tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] \
                                              - (1 / 3 * tildext1_N[i, :] ** 3 - tildext1_N[i, :] + tildext1_N_y[i,
                                                                                                    :]) * dt \
                                              - alpha_t[i] * ((tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 0,
                                                                                            :] + tildeyt1_N_y[i, 0,
                                                                                                 :]) * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
                    tildeyt1_N_y[i + 1, 0, :] = tildeyt1_N_y[i, 0, :] + tildeyt1_N[i + 1, 0, :] * dt

            else:
                for j in range(N):
                    tildeyt1_N[i + 1, 0, j] = tildeyt1_N[i, 0, j] \
                                              - grad_theta_grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 0, j] * dt \
                                              - 1 / N * np.sum(tildeyt1_N[i, 0, j] * grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[1, 0, :] * - grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt

        if est_beta:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                              - (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt
                if fitzhugh:
                    tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                              - alpha_t[i] * ((tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 1,
                                                                                            :] + tildeyt1_N_y[i, 1,
                                                                                                 :]) * dt \
                                              - beta_t[i] * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                              - (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt
                    tildeyt1_N_y[i + 1, 1, :] = tildeyt1_N_y[i, 1, :] + tildeyt1_N[i + 1, 1, :] * dt

            else:
                for j in range(N):
                    tildeyt1_N[i + 1, 1, j] = tildeyt1_N[i, 1, j] \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 1, j] * dt \
                                              - 1 / N * np.sum(grad_theta_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[i, 1, j] * grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(tildeyt1_N[1, 1, :] * - grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt

        if est_gamma:
            tildeyt1_N[i + 1, 2, :] = tildeyt1_N[i, 2, :] \
                                      - alpha_t[i] * (
                                                  (tildext1_N[i, :] ** 2 - 1) * tildeyt1_N[i, 2, :] + tildeyt1_N_y[i, 2,
                                                                                                      :]) * dt \
                                      - beta_t[i] * (tildeyt1_N[i, 2, :] - np.mean(tildeyt1_N[i, 2, :])) * dt
            tildeyt1_N_y[i + 1, 2, :] = tildeyt1_N_y[i, 2, :] + (1 + tildeyt1_N[i + 1, 2, :]) * dt

        # alpha_t
        if est_alpha:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    alpha_t[i+1] = alpha_t[i] + gamma \
                                   * (- grad_theta_grad_v(xt[i], alpha_t[i]) + beta_t[i] * tildeyt1_N[i, 0, 0]) \
                                   * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - tildext2_N[i, 0])) * dt)
                if fitzhugh:
                    alpha_t[i+1] = alpha_t[i] + gamma * (-(1/3*xt[i]**3 - xt[i] + yt[i]) + beta_t[i] * tildeyt1_N[i, 0, 0]) \
                                   * (dxt - (- alpha_t[i] * (1/3*xt[i]**3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - tildext2_N[i, 0])) * dt)
            else:
                alpha_t[i+1] = alpha_t[i+1] = alpha_t[i] + gamma \
                               * (- grad_theta_grad_v(xt[i], alpha_t[i]) + tildeyt1_N[i, 0, 0] * grad_x_grad_w(xt[i] - tildext1_N[i, 0], beta_t[i])) \
                               * 1/sigma**2 * (dxt - (-grad_v(xt[i], alpha_t[i]) - grad_w(xt[i] - tildext2_N[i, 0], beta_t[i])) * dt)

        # beta_t
        if est_beta:
            if grad_w == grad_quadratic:
                if not fitzhugh:
                    beta_t[i + 1] = beta_t[i] + gamma \
                                    * (- (xt[i] - tildext1_N[i, 0]) + beta_t[i] * tildeyt1_N[i, 1, 0]) \
                                    * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - tildext2_N[i, 0])) * dt)
                if fitzhugh:
                    beta_t[i+1] = beta_t[i] + gamma \
                                  * (- (xt[i] - tildext1_N[i, 0]) + beta_t[i] * tildeyt1_N[i, 1, 0]) \
                                  * (dxt - (- alpha_t[i] * (1/3*xt[i]**3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - tildext2_N[i, 0])) * dt)

            else:
                beta_t[i + 1] = beta_t[i] + gamma \
                                * (- grad_theta_grad_w(xt[i] - tildext1_N[i, 0], beta_t[i]) + grad_x_grad_w(xt[i] - tildext1_N[i, 0], beta_t[i]) * tildeyt1_N[i, 1, 0]) \
                                * 1/sigma**2 * (dxt - (-grad_v(xt[i], alpha_t[i]) - grad_w(xt[i] - tildext2_N[i, 0], beta_t[i])) * dt)

        # gamma_t
        if est_gamma:
            gamma_t[i+1] = gamma_t[i] \
                           + gamma \
                           * (beta_t[i] * tildeyt1_N[i, 2, 0])\
                           * (dxt - (- alpha_t[i] * (1/3*xt[i]**3 - xt[i] + yt[i]) - beta_t[i] * (xt[i] - tildext2_N[i, 0])) * dt) \
                           + gamma \
                           * (dyt - (gamma_t[i] + xt[i]) * dt)

    if not fitzhugh:
        return alpha_t, beta_t
    if fitzhugh:
        return alpha_t, beta_t, gamma_t

#######################


if __name__ == "__main__":

    # general
    root = "results/"
    leaf = "kuramoto"
    path = os.path.join(root, leaf)
    if not os.path.exists(path):
        os.makedirs(path)

    # simulation parameters
    N_obs = 1
    N_par = 100
    T = 1000
    dt = 0.1
    alpha = 1.0
    grad_v = grad_quadratic
    grad_theta_grad_v = grad_theta_grad_quadratic
    grad_x_grad_v = grad_x_grad_quadratic
    beta = 0.8
    grad_w = grad_kuramoto
    grad_theta_grad_w = grad_theta_grad_kuramoto
    grad_x_grad_w = grad_x_grad_kuramoto
    sigma = .1
    kuramoto = True
    fitzhugh = False
    seeds = range(10)

    nt = round(T / dt)
    t = [i * dt for i in range(nt + 1)]

    # estimation parameters
    gamma = 0.05

    alpha0 = 2.0
    alpha_true = alpha
    est_alpha = False

    beta0 = 0.2
    beta_true = beta
    est_beta = True

    gamma0 = 1.0
    gamma_true = 0.3
    est_gamma = False

    N_est = 20

    # plotting
    plot_each_run = False
    plot_mean_run = True

    # output
    save_plots = True

    all_alpha_t = np.zeros((nt + 1, len(seeds), 2))
    all_beta_t = np.zeros((nt + 1, len(seeds), 2))

    if fitzhugh:
        all_gamma_t = np.zeros((nt + 1, len(seeds), 2))

    for idx, seed in enumerate(seeds):

        print(seed)

        # simulate mvsde
        x0 = np.random.uniform(N_par)
        y0 = np.random.uniform(N_par)

        if not fitzhugh:
            xt = sde_sim_func(N_par, T, grad_v, alpha_true, grad_w, beta_true, sigma=sigma, x0=x0, dt=dt, seed=seed,
                              kuramoto=kuramoto)
        if fitzhugh:
            xt, yt = sde_sim_func(N_par, T, grad_v, alpha, grad_w, beta, sigma=sigma, x0=x0, dt=dt, seed=seed,
                                  fitzhugh=True, y0=y0, gamma=gamma_true)

        if not fitzhugh:
            alpha_t_one, beta_t_one = online_est_one(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true,
                                                     est_alpha, grad_w, grad_theta_grad_w, grad_x_grad_w, beta0,
                                                     beta_true, est_beta, sigma, gamma, N_est, seed, kuramoto=kuramoto)
            alpha_t_two, beta_t_two = online_est_two(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true,
                                                     est_alpha, grad_w, grad_theta_grad_w, grad_x_grad_w, beta0,
                                                     beta_true, est_beta, sigma, gamma, N_est, seed, kuramoto=kuramoto)
        if fitzhugh:
            alpha_t_one, beta_t_one, gamma_t_one = online_est_one(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0,
                                                                  alpha_true, est_alpha, grad_w, grad_theta_grad_w,
                                                                  grad_x_grad_w, beta0, beta_true, est_beta, sigma,
                                                                  gamma, N_est, seed, fitzhugh, yt, gamma0, gamma_true,
                                                                  est_gamma)
            alpha_t_two, beta_t_two, gamma_t_two = online_est_two(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0,
                                                                  alpha_true, est_alpha, grad_w, grad_theta_grad_w,
                                                                  grad_x_grad_w, beta0, beta_true, est_beta, sigma,
                                                                  gamma, N_est, seed, fitzhugh, yt, gamma0, gamma_true,
                                                                  est_gamma)


        all_alpha_t[:, idx, 0], all_beta_t[:, idx, 0] = alpha_t_one, beta_t_one
        all_alpha_t[:, idx, 1], all_beta_t[:, idx, 1] = alpha_t_two, beta_t_two

        if fitzhugh:
            all_gamma_t[:, idx, 0] = gamma_t_one
            all_gamma_t[:, idx, 1] = gamma_t_two

        if plot_each_run:
            if est_alpha and not est_beta and not est_gamma:
                plt.plot(t, alpha_t_one, label=r"$\alpha_{t}^N$ (Estimator 1)", color="C0")
                plt.plot(t, alpha_t_two, label=r"$\alpha_{t}^N$ (Estimator 2)", color="C0")
                plt.axhline(y=alpha, linestyle="--", color="C1")
                plt.legend()
                plt.show()
            if est_beta and not est_alpha and not est_gamma:
                plt.plot(t, beta_t_one, label=r"$\beta_{t}^N$ (Estimator 1)", color="C0")
                plt.plot(t, beta_t_two, label=r"$\beta_{t}^N$ (Estimator 2)", color="C0")
                plt.axhline(y=beta, linestyle="--", color="C1")
                plt.legend()
                plt.show()
            if est_gamma and not est_alpha and not est_beta:
                plt.plot(t, gamma_t_one, label=r"$\gamma_{t}^N$ (Estimator 1)", color="C0")
                plt.plot(t, gamma_t_two, label=r"$\gamma_{t}^N$ (Estimator 2)", color="C0")
                plt.axhline(y=beta, linestyle="--", color="C1")
                plt.legend()
                plt.show()


    if plot_mean_run:
        if est_alpha and not est_beta and not est_gamma:
            plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
            plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (Estimator 2)")
            plt.axhline(y=alpha, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(path + "/alpha_est_all.eps", dpi=300)
            plt.show()
        elif est_beta and not est_alpha and not est_gamma:
            plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (Estimator 1)")
            plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (Estimator 2)")
            plt.axhline(y=beta, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(path + "/beta_est_all_ex2.eps", dpi=300)
            plt.show()
        elif est_gamma and not est_alpha and not est_beta:
            plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (Estimator 1)")
            plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (Estimator 2)")
            plt.axhline(y=gamma_true, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(path + "/gamma_est_all.eps", dpi=300)
            plt.show()