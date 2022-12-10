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

def sde_sim_func(N=20, T=100, grad_v=grad_quadratic, alpha=0.1,
                 grad_w=grad_quadratic, beta=None, Aij=None,
                 Lij=None, sigma=1, x0=1, dt=0.1, seed=1,
                 Aij_calc=False, Aij_calc_func=None,
                 Aij_influence_func=None, Aij_scale_param=1,
                 kuramoto=False, **kwargs):
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

    # if in simple mean field form
    if beta is not None:
        if grad_w == grad_quadratic:
            for i in tqdm(range(0, nt)):
                xt[i + 1, :] = xt[i, :] - grad_v(xt[i, :], alpha[i]) * dt - beta[i] * (
                            xt[i, :] - np.mean(xt[i, :])) * dt + sigma * dwt[i, :]
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
                   grad_theta_grad_w, grad_x_grad_w, beta0, beta_true, est_beta, sigma, gamma, N=2, seed=1):

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

        # tilde x_t1^N and tilde x_t2^N
        if grad_w == grad_quadratic:
            tildext1_N[i + 1, :] = tildext1_N[i, :] - grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                   - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                   + sigma * dwt1[i, :]
            tildext2_N[i + 1, :] = tildext2_N[i, :] - grad_v(tildext2_N[i, :], alpha_t[i]) * dt \
                                   - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                   + sigma * dwt2[i, :]

        else:
            for j in range(N):
                tildext1_N[i + 1, j] = tildext1_N[i, j] - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt1[i, j]
                tildext2_N[i + 1, j] = tildext2_N[i, j] - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt2[i, j]

        # tilde y_t1^N
        if est_alpha:
            if grad_w == grad_quadratic:
                tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] - grad_theta_grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                          - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                          - beta_t[i] * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
            else:
                tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] ## to do

        if est_beta:
            if grad_w == grad_quadratic:
                tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                          - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                          - beta_t[i] * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                          + np.mean(tildext1_N[i, :]) * dt
            else:
                tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] ## to do


        # alpha_t
        if est_alpha:
            if grad_w == grad_quadratic:
                alpha_t[i+1] = alpha_t[i] + gamma \
                               * (- grad_theta_grad_v(xt[i], alpha_t[i]) + beta_t[i] * np.mean(tildeyt1_N[i, 0, :])) \
                               * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)
            else:
                alpha_t[i+1] = alpha_t[i] ## to do

        # beta_t
        if est_beta:
            if grad_w == grad_quadratic:
                beta_t[i + 1] = beta_t[i] + gamma \
                                * (- (xt[i] - np.mean(tildext1_N[i, :])) + beta_t[i] * np.mean(tildeyt1_N[i, 1, :])) \
                                * (dxt - (-grad_v(xt[i], alpha_t[i]) - beta_t[i] * (xt[i] - np.mean(tildext2_N[i, :]))) * dt)
            else:
                alpha_t[i + 1] = alpha_t[i] ## to do

    return alpha_t, beta_t

#######################


if __name__ == "__main__":

    # general
    root = "results/"
    leaf = "bistable_v_quadratic_w"
    path = os.path.join(root, leaf)
    if not os.path.exists(path):
        os.makedirs(path)

    # simulation parameters
    N_obs = 1
    N_par = 200
    T = 10000
    dt = 0.1
    alpha = 1.5
    grad_v = grad_bi_stable
    grad_theta_grad_v = grad_theta_grad_bi_stable
    grad_x_grad_v = grad_x_grad_bi_stable
    beta = 0.7
    grad_w = grad_quadratic
    grad_theta_grad_w = grad_theta_grad_quadratic
    grad_x_grad_w = grad_x_grad_quadratic
    sigma = 1
    seeds = range(5)

    nt = round(T / dt)
    t = [i * dt for i in range(nt + 1)]

    # estimation parameters
    gamma = 0.005

    alpha0 = 2.0
    alpha_true = alpha
    est_alpha = False

    beta0 = 0.3
    beta_true = beta
    est_beta = True

    N_est = 100

    # plotting
    plot_each_run = False
    plot_mean_run = True

    # output
    save_plots = True

    all_alpha_t = np.zeros((nt + 1, len(seeds)))
    all_beta_t = np.zeros((nt + 1, len(seeds)))

    for idx, seed in enumerate(seeds):

        print(seed)

        # simulate mvsde
        x0 = np.random.randn(1)
        xt = sde_sim_func(N_par, T, grad_v, alpha, grad_w, beta, sigma=sigma, x0=x0, dt=dt, seed=seed)

        alpha_t, beta_t = online_est_one(xt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true, est_alpha,
                                         grad_w, grad_theta_grad_w, grad_x_grad_w, beta0, beta_true, est_beta, sigma,
                                         gamma, N_est, seed)

        all_alpha_t[:, idx], all_beta_t[:, idx] = alpha_t, beta_t

        if plot_each_run:
            if est_alpha and not est_beta:
                plt.plot(t, alpha_t, label=r"$\alpha_{t}^N$ (Estimator 1)", color="C0")
                plt.axhline(y=alpha, linestyle="--", color="C1")
                plt.legend()
                plt.show()
            if est_beta and not est_alpha:
                plt.plot(t, beta_t, label=r"$\beta_{t}^N$ (Estimator 1)", color="C0")
                plt.axhline(y=beta, linestyle="--", color="C1")
                plt.legend()
                plt.show()

    if plot_mean_run:
        if est_alpha and not est_beta:
            plt.plot(t, np.mean(all_alpha_t, 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
            plt.axhline(y=alpha, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(path + "/alpha_est_all.eps", dpi=300)
            plt.show()
        elif est_beta and not est_alpha:
            plt.plot(t, np.mean(all_beta_t, 1), label=r"$\beta_{t}^N$ (Estimator 1)")
            plt.axhline(y=beta, linestyle="--", color="black")
            plt.legend()
            if save_plots:
                plt.savefig(path + "/beta_est_all.eps", dpi=300)
            plt.show()