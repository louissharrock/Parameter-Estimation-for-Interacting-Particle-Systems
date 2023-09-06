import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


#############################
#### POTENTIAL FUNCTIONS ####
#############################

## Gradient of quadratic potential (confinement or interaction)
def grad_quadratic(x, alpha1, alpha2=None):
    return alpha1 * x


def grad_theta_grad_quadratic(x, alpha1, alpha2=None):
    return x


def grad_x_grad_quadratic(x, alpha1, alpha2=None):
    return alpha1


## Gradient of linear potential (null) (confinement or interaction)
def grad_linear(x, alpha1, alpha2=None):
    return 0*x


def grad_theta_grad_linear(x, alpha1, alpha2=None):
    return 0*x


def grad_x_grad_linear(x, alpha1, alpha2=None):
    return 0*x


## Gradient of bi-stable (Landau) potential (confinement or interaction)
def grad_bi_stable(x, alpha1, alpha2=None):
    return alpha1 * (x ** 3 - x)


def grad_theta_grad_bi_stable(x, alpha1, alpha2=None):
    return x**3 - x


def grad_x_grad_bi_stable(x, alpha1, alpha2=None):
    return alpha1 * (3 * x ** 2 - 1)


## Gradient of sine (Kuramoto) potential (interaction)
def grad_kuramoto(x, alpha1, alpha2=None):
    return alpha1 * np.sin(x)


def grad_theta_grad_kuramoto(x, alpha1, alpha2=None):
    return np.sin(x)


def grad_x_grad_kuramoto(x, alpha1, alpha2=None):
    return alpha1 * np.cos(x)


## Fitzhugh-Nagumo potential (confinement)
def grad_fitzhugh(x, y, alpha, alpha2=None):
    return alpha * (1 / 3 * x ** 3 - x + y)


def grad_theta_grad_fitzhugh(x, y, alpha, alpha2=None):
    return 1 / 3 * x ** 3 - x + y


def grad_x1_grad_fitzhugh(x, y, alpha, alpha2=None):
    return alpha * (x ** 2 - 1)


def grad_x2_grad_fitzhugh(x, y, alpha, alpha2=None):
    return alpha


## Cucker-Smale potential (interaction)
def grad_cucker_smale(x, v, alpha1, alpha2):
    num = alpha1 * v
    denom = (1 + x ** 2) ** alpha2
    return num / denom


def grad_theta1_grad_cucker_smale(x, v, alpha1, alpha2):
    num = v
    denom = (1 + x ** 2) ** alpha2
    return num / denom


def grad_theta2_grad_cucker_smale(x, v, alpha1, alpha2):
    num = - alpha1 * np.log((1 + x ** 2)) * v
    denom = (1 + x ** 2) ** alpha2
    return num / denom


def grad_x1_grad_cucker_smale(x, v, alpha1, alpha2):
    num = - 2 * alpha1 * alpha2 * (x) * v
    denom = (1 + x ** 2) ** (alpha2 + 1)
    return num / denom


def grad_x2_grad_cucker_smale(x, v, alpha1, alpha2):
    num = alpha1
    denom = (1 + x ** 2) ** alpha2
    return num / denom


## Stochastic volatility potential
def grad_stochastic_volatility(x, alpha1, alpha2):
    return x * (alpha1 * x - alpha2)


def grad_theta1_grad_stochastic_volatility(x, alpha1, alpha2):
    return x * x


def grad_theta2_grad_stochastic_volatility(x, alpha1, alpha2):
    return - x


def grad_x_grad_stochastic_volatility(x, alpha1, alpha2):
    return 2 * alpha1 * x - alpha2


## Stochastic opinion dynamics

EPS = .1

def opinion_dynamics_kernel(r, alpha):
    if -alpha < r < alpha:
        return np.exp(EPS - EPS / (1 - (r / alpha) ** 2))
    else:
        return 0


def grad_opinion_dynamics(x, alpha1, alpha2):
    r = abs(x)
    if -alpha2 < r < alpha2:
        return alpha1 * opinion_dynamics_kernel(r, alpha2) * x
    else:
        return 0


def grad_theta1_grad_opinion_dynamics(x, alpha1, alpha2):
    r = abs(x)
    if -alpha2 < r < alpha2:
        return opinion_dynamics_kernel(r, alpha2) * x
    else:
        return 0


def grad_theta2_grad_opinion_dynamics(x, alpha1, alpha2):
    r = abs(x)
    if -alpha2 < r < alpha2:
        return 2 * EPS * alpha1 / (alpha2 ** 3) * r ** 2 * 1 / ((1 - r ** 2 / (alpha2 ** 2)) ** 2) * opinion_dynamics_kernel(r, alpha2) * x
    else:
        return 0


## gradient of double sine potential
def grad_double_kuramoto(x, alpha1, alpha2):
    return alpha1 * np.sin(x) + alpha2 * np.sin(2*x)


def grad_theta_grad_double_kuramoto(x, alpha1, alpha2):
    return np.sin(x)


def grad_theta2_grad_double_kuramoto(x, alpha1, alpha2):
    return np.sin(2*x)


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


def sde_sim_func(N=20, T=100, grad_v=grad_quadratic, alpha=1, grad_w=grad_quadratic, beta=0.1, Aij=None, Lij=None,
                 sigma=1, x0=1, dt=0.1, seed=1, kuramoto=False, fitzhugh=False, y0=None, gamma=None, cucker_smale=False,
                 v0=None, beta2=None, stochastic_volatility=False, alpha2=None, all_particles=False,
                 opinion_dynamics=False, double_kuramoto=False):

    # check inputs
    if fitzhugh:
        assert y0 is not None
        assert gamma is not None
        assert grad_v == grad_fitzhugh
        assert grad_w == grad_quadratic

    if cucker_smale:
        assert v0 is not None
        assert beta2 is not None
        assert grad_v == grad_quadratic
        assert grad_w == grad_cucker_smale

    if kuramoto:
        assert grad_w == grad_kuramoto

    if stochastic_volatility:
        assert alpha2 is not None
        assert grad_v == grad_stochastic_volatility
        assert grad_w == grad_quadratic

    if opinion_dynamics:
        assert beta2 is not None
        assert grad_v == grad_linear
        assert grad_w == grad_opinion_dynamics

    if double_kuramoto:
        assert beta2 is not None
        assert grad_w == grad_double_kuramoto

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = int(np.round(T / dt))

    # parameters
    if type(alpha) is float or type(alpha) is int:
        alpha = [alpha] * (nt + 1)

    if type(beta) is float or type(beta) is int:
        beta = [beta] * (nt + 1)

    if fitzhugh:
        if type(gamma) is int or type(gamma) is float:
            gamma = [gamma] * (nt+1)

    if cucker_smale or opinion_dynamics or double_kuramoto:
        if type(beta2) is int or type(beta2) is float:
            beta2 = [beta2] * (nt+1)

    if stochastic_volatility:
        if type(alpha2) is int or type(alpha2) is float:
            alpha2 = [alpha2] * (nt+1)

    if stochastic_volatility:
        if type(sigma) is int or type(sigma) is float:
            sigma = [sigma] * (nt+1)

    # initialise xt
    xt = np.zeros((nt + 1, N))
    xt[0, :] = x0

    # intialise yt
    if fitzhugh:
        yt = np.zeros((nt+1, N))
        yt[0, :] = y0

    # initialise vt
    if cucker_smale:
        vt = np.zeros((nt+1, N))
        vt[0, :] = v0

    # brownian motion
    dwt = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # simulate

    # if interaction parameter provided
    if beta is not None:

        if fitzhugh:
            for i in tqdm(range(0, nt)):
                xt[i + 1, :] = xt[i, :] \
                               - grad_v(xt[i, :], yt[i, :], alpha[i]) * dt \
                               - grad_w(xt[i] - np.mean(xt[i, :]), beta[i]) * dt \
                               + sigma * dwt[i, :]
                yt[i + 1, :] = yt[i, :] + (gamma[i] + xt[i, :]) * dt
        elif cucker_smale:
            for i in tqdm(range(0, nt)):
                xt[i + 1, :] = xt[i, :] + vt[i, :] * dt
                for j in range(N):
                    vt[i + 1, j] = vt[i, j] \
                                   - grad_v(xt[i, j], alpha[i]) * dt \
                                   - 1 / N * np.sum(grad_w(xt[i, j] - xt[i, :], vt[i, j] - vt[i, :], beta[i], beta2[i])) * dt \
                                   + sigma * dwt[i, j]
        elif kuramoto:
            for i in tqdm(range(0, nt)):
                for j in range(N):
                    xt[i + 1, j] = xt[i, j] \
                                   - grad_v(xt[i, j], alpha[i]) * dt \
                                   - 1 / N * np.sum(grad_w(xt[i, j] - xt[i, :], beta[i])) * dt \
                                   + sigma * dwt[i, j]
                while np.any(xt[i + 1, :] > + np.pi) or np.any(xt[i + 1, :] < - np.pi):
                    xt[i + 1, np.where(xt[i + 1, :] > +np.pi)] -= 2. * np.pi
                    xt[i + 1, np.where(xt[i + 1, :] < -np.pi)] += 2. * np.pi
        elif double_kuramoto:
            for i in tqdm(range(0, nt)):
                for j in range(N):
                    xt[i + 1, j] = xt[i, j] \
                                   - grad_v(xt[i, j], alpha[i]) * dt \
                                   - 1 / N * np.sum(grad_w(xt[i, j] - xt[i, :], beta[i], beta2[i])) * dt \
                                   + sigma * dwt[i, j]
                while np.any(xt[i + 1, :] > + np.pi) or np.any(xt[i + 1, :] < - np.pi):
                    xt[i + 1, np.where(xt[i + 1, :] > +np.pi)] -= 2. * np.pi
                    xt[i + 1, np.where(xt[i + 1, :] < -np.pi)] += 2. * np.pi
        elif stochastic_volatility:
            for i in tqdm(range(0, nt)):
                xt[i + 1, :] = xt[i, :] \
                               - grad_v(xt[i, :], alpha[i], alpha2[i]) * dt \
                               - grad_w(xt[i, :] - np.mean(xt[i, :]), beta[i]) * dt \
                               + sigma[i] * xt[i, :] ** 1.5 * dwt[i, :]
        elif opinion_dynamics:
            for i in tqdm(range(0, nt)):
                for j in range(N):
                    xt[i + 1, j] = xt[i, j] \
                               - grad_v(xt[i, j], alpha[i]) * dt \
                               - 1 / N * np.sum(np.vectorize(grad_w, otypes=[np.float64])(xt[i, j] - xt[i, :], beta[i], beta2[i])) * dt \
                               + sigma * dwt[i, j]

        else:
            if grad_w == grad_quadratic:
                for i in tqdm(range(0, nt)):
                    xt[i + 1, :] = xt[i, :] \
                                   - grad_v(xt[i, :], alpha[i]) * dt \
                                   - grad_w(xt[i, :] - np.mean(xt[i, :]), beta[i]) * dt \
                                   + sigma * dwt[i, :]
            if grad_w != grad_quadratic:
                for i in tqdm(range(0, nt)):
                    for j in range(N):
                        xt[i + 1, j] = xt[i, j] \
                                       - grad_v(xt[i, j], alpha[i]) * dt \
                                       - 1 / N * np.sum(grad_w(xt[i, j] - xt[i, :], beta[i])) * dt \
                                       + sigma * dwt[i, j]

    # if in Aij form
    if np.any(Aij):
        for i in tqdm(range(0, nt)):
            for j in range(0, N):
                xt[i + 1, j] = xt[i, j] - grad_v(xt[i, j], alpha[i]) * dt \
                               - np.dot(Aij[j, :],xt[i, j] - xt[i, :]) * dt + sigma * dwt[i, j]

    # if in Lij form
    if np.any(Lij):
        for i in tqdm(range(0, nt)):
            xt[i + 1, :] = xt[i, :] - grad_v(xt[i, :], alpha[i]) * dt - np.dot(Lij, xt[i, :]) * dt + sigma * dwt[i, :]

    if all_particles:
        if fitzhugh:
            return xt, yt
        elif cucker_smale:
            return xt, vt
        else:
            return xt
    else:
        if fitzhugh:
            return xt[:, 0], yt[:, 0]
        elif cucker_smale:
            return xt[:, 0], vt[:, 0]
        else:
            return xt[:, 0]

#######################


##############################################
#### ONLINE ESTIMATORS (MVSDE LIKELIHOOD) ####
##############################################

## Recursive MLE

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
## -> seed (random seed)
## -> whether to compute average over particles (estimator 1) or not (estimator 2)

## Outputs:
## -> theta_t (online parameter estimate)

def online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true, est_alpha, grad_w,
               grad_theta_grad_w, grad_x_grad_w, beta0, beta_true, est_beta, sigma, gamma, N=2, seed=1, average=True,
               fitzhugh=False, yt=None, grad_x2_grad_v=None, gamma0=None, gamma_true=None, est_gamma=False,
               kuramoto=False, cucker_smale=False, vt=None, grad_theta2_grad_w=None, grad_x2_grad_w=None, beta20=None,
               beta2_true=None, est_beta2=False, stochastic_volatility=False, grad_theta2_grad_v=None, alpha20=None,
               alpha2_true=None, est_alpha2=False, sigma0=None, sigma_true=None, est_sigma=False):

    # check inputs
    if fitzhugh:
        assert yt is not None
        assert gamma0 is not None
        assert gamma_true is not None
        assert grad_v == grad_fitzhugh
        assert grad_theta_grad_v == grad_theta_grad_fitzhugh
        assert grad_x_grad_v == grad_x1_grad_fitzhugh
        assert grad_x2_grad_v == grad_x2_grad_fitzhugh
        assert grad_w == grad_quadratic
        assert grad_theta_grad_w == grad_theta_grad_quadratic
        assert grad_x_grad_w == grad_x_grad_quadratic

    if cucker_smale:
        assert vt is not None
        assert beta20 is not None
        assert beta2_true is not None
        assert grad_v == grad_quadratic
        assert grad_theta_grad_v == grad_theta_grad_quadratic
        assert grad_x_grad_v == grad_x_grad_quadratic
        assert grad_w == grad_cucker_smale
        assert grad_theta_grad_w == grad_theta1_grad_cucker_smale
        assert grad_theta2_grad_w == grad_theta2_grad_cucker_smale
        assert grad_x_grad_w == grad_x1_grad_cucker_smale
        assert grad_x2_grad_w == grad_x2_grad_cucker_smale

    if kuramoto:
        assert grad_w == grad_kuramoto
        assert grad_theta_grad_w == grad_theta_grad_kuramoto
        assert grad_x_grad_w == grad_x_grad_kuramoto

    if stochastic_volatility:
        assert alpha20 is not None
        assert alpha2_true is not None
        assert sigma0 is not None
        assert sigma_true is not None
        assert grad_v == grad_stochastic_volatility
        assert grad_theta_grad_v == grad_theta1_grad_stochastic_volatility
        assert grad_theta2_grad_v == grad_theta2_grad_stochastic_volatility
        assert grad_x_grad_v == grad_x_grad_stochastic_volatility
        assert grad_w == grad_quadratic
        assert grad_theta_grad_w == grad_theta_grad_quadratic
        assert grad_x_grad_w == grad_x_grad_quadratic

    # averaging func
    if average:
        def averaging_func(x):
            return np.mean(x)
    if not average:
        def averaging_func(x):
            return x[1]

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = xt.shape[0] - 1

    # learning rate
    if type(gamma) is float or type(gamma) is int:
        all_gamma = [gamma] * (nt + 1)
    else:
        all_gamma = gamma

    # parameters
    if type(alpha_true) is float or type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float or type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    if fitzhugh:
        if type(gamma_true) is float or type(gamma_true) is int:
            gamma_true = [gamma_true] * (nt + 1)

    if cucker_smale:
        if type(beta2_true) is float or type(beta2_true) is int:
            beta2_true = [beta2_true] * (nt + 1)

    if stochastic_volatility:
        if type(alpha2_true) is float or type(alpha2_true) is int:
            alpha2_true = [alpha2_true] * (nt + 1)

        if type(sigma_true) is float or type(sigma_true) is int:
            sigma_true = [sigma_true] * (nt + 1)

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

    gamma_t = np.zeros(nt + 1)
    if fitzhugh:
        if est_gamma:
            gamma_t[0] = gamma0
        else:
            gamma_t = gamma_true

    beta2_t = np.zeros(nt + 1)
    if cucker_smale:
        if est_beta2:
            beta2_t[0] = beta20
        else:
            beta2_t = beta2_true

    alpha2_t = np.zeros(nt + 1)
    if stochastic_volatility:
        if est_alpha2:
            alpha2_t[0] = alpha20
        else:
            alpha2_t = alpha2_true

        sigma_t = np.zeros(nt + 1)
        if est_sigma:
            sigma_t[0] = sigma0
        else:
            sigma_t = sigma_true

    tildext1_N = np.zeros((nt + 1, N))
    tildext2_N = np.zeros((nt + 1, N))
    tildext1_N[0, :] = np.random.normal(0,1,N) # xt[0] * np.ones(N)
    tildext2_N[0, :] = np.random.normal(0,1,N) # xt[0] * np.ones(N)

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

    if cucker_smale:
        tildext1_N_v = np.zeros((nt + 1, N))
        tildext2_N_v = np.zeros((nt + 1, N))
        tildext1_N_v[0, :] = np.random.normal(0,1,N)
        tildext2_N_v[0, :] = np.random.normal(0,1,N)

        tildeyt1_N = np.zeros((nt + 1, 3, N)) # tangent of x process, three parameters
        tildeyt1_N[0, :] = np.zeros((3, N))

        tildeyt1_N_v = np.zeros((nt + 1, 3, N)) # tangent of v process, three parameters
        tildeyt1_N_v[0, :] = np.zeros((3, N))

    if stochastic_volatility:
        tildeyt1_N = np.zeros((nt + 1, 3, N)) # tangent of x process w.r.t. theta, three parameters
        tildeyt1_N[0, :] = np.zeros((3, N)) # tangent of x process w.r.t. theta, three parameters

    dwt1 = np.sqrt(dt) * np.random.randn(nt + 1, N)
    dwt2 = np.sqrt(dt) * np.random.randn(nt + 1, N)

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xt[i + 1] - xt[i]

        if fitzhugh:
            dyt = yt[i+1] - yt[i]

        if cucker_smale:
            dvt = vt[i+1] - vt[i]

        # IPS integrated with parameters
        if fitzhugh:
            tildext1_N[i + 1, :] = tildext1_N[i, :] \
                           - grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * dt \
                           - grad_w(tildext1_N[i] - np.mean(tildext1_N[i, :]), beta_t[i]) * dt \
                           + sigma * dwt1[i, :]
            tildext1_N_y[i + 1, :] = tildext1_N_y[i, :] + (gamma_t[i] + tildext1_N[i, :]) * dt

            tildext2_N[i + 1, :] = tildext2_N[i, :] \
                                   - grad_v(tildext2_N[i, :], tildext2_N_y[i, :], alpha_t[i]) * dt \
                                   - grad_w(tildext2_N[i] - np.mean(tildext2_N[i, :]), beta_t[i]) * dt \
                                   + sigma * dwt2[i, :]
            tildext2_N_y[i + 1, :] = tildext2_N_y[i, :] + (gamma_t[i] + tildext2_N[i, :]) * dt

        elif cucker_smale:
            tildext1_N[i + 1, :] = tildext1_N[i, :] + tildext1_N_v[i, :] * dt
            for j in range(N):
                tildext1_N_v[i + 1, j] = tildext1_N_v[i, j] \
                                         - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                         - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i])) * dt \
                                         + sigma * dwt1[i, j]
            tildext2_N[i + 1, :] = tildext2_N[i, :] + tildext2_N_v[i, :] * dt
            for j in range(N):
                tildext2_N_v[i + 1, j] = tildext2_N_v[i, j] \
                                         - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                         - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], tildext2_N_v[i, j] - tildext2_N_v[i, :], beta_t[i], beta2_t[i])) * dt \
                                         + sigma * dwt2[i, j]
        elif kuramoto:
            for j in range(N):
                tildext1_N[i + 1, j] = tildext1_N[i, j] \
                                       - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt1[i, j]
                while np.any(tildext1_N[i + 1, :] > + np.pi) or np.any(tildext1_N[i + 1, :] < - np.pi):
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                    tildext1_N[i + 1, np.where(tildext1_N[i + 1, :] < -np.pi)] += 2. * np.pi

                tildext2_N[i + 1, j] = tildext2_N[i, j] \
                                       - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                       - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], beta_t[i])) * dt \
                                       + sigma * dwt2[i, j]
            while np.any(tildext2_N[i + 1, :] > + np.pi) or np.any(tildext2_N[i + 1, :] < - np.pi):
                tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] > +np.pi)] -= 2. * np.pi
                tildext2_N[i + 1, np.where(tildext2_N[i + 1, :] < -np.pi)] += 2. * np.pi

        elif stochastic_volatility:
            tildext1_N[i + 1, :] = tildext1_N[i, :] \
                                   - grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * dt \
                                   - grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * dt \
                                   + sigma_t[i] * tildext1_N[i, :] ** 1.5 * dwt1[i, :]
            tildext2_N[i + 1, :] = tildext2_N[i, :] \
                                   - grad_v(tildext2_N[i, :], alpha_t[i], alpha2_t[i]) * dt \
                                   - grad_w(tildext2_N[i, :] - np.mean(tildext2_N[i, :]), beta_t[i]) * dt \
                                   + sigma_t[i] * tildext1_N[i, :] ** 1.5 * dwt2[i, :]

        else:
            if grad_w == grad_quadratic:
                tildext1_N[i + 1, :] = tildext1_N[i, :] \
                                       - grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext1_N[i, :] - np.mean(tildext1_N[i, :])) * dt \
                                       + sigma * dwt1[i, :]
                tildext2_N[i + 1, :] = tildext2_N[i, :] - grad_v(tildext2_N[i, :], alpha_t[i]) * dt \
                                       - beta_t[i] * (tildext2_N[i, :] - np.mean(tildext2_N[i, :])) * dt \
                                       + sigma * dwt2[i, :]

            if grad_w != grad_quadratic:
                for j in range(N):
                    tildext1_N[i + 1, j] = tildext1_N[i, j] \
                                           - grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                           - 1 / N * np.sum(grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                           + sigma * dwt1[i, j]
                    tildext2_N[i + 1, j] = tildext2_N[i, j] \
                                           - grad_v(tildext2_N[i, j], alpha_t[i]) * dt \
                                           - 1 / N * np.sum(grad_w(tildext2_N[i, j] - tildext2_N[i, :], beta_t[i])) * dt \
                                           + sigma * dwt2[i, j]

        # tangent IPS integrated with parameters
        if est_alpha:
            if fitzhugh:
                tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] \
                                          - grad_theta_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * dt \
                                          - grad_x_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                          - grad_x2_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N_y[i, 0, :] * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt
                tildeyt1_N_y[i + 1, 0, :] = tildeyt1_N_y[i, 0, :] + tildeyt1_N[i + 1, 0, :] * dt
            elif cucker_smale:
                tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] + tildeyt1_N_v[i, 0, :] * dt
                for j in range(N):
                    tildeyt1_N_v[i + 1, 0, j] = tildeyt1_N_v[i, 0, j] \
                                                - grad_theta_grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                                - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 0, j] * dt \
                                                - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :],beta_t[i], beta2_t[i]) * (tildeyt1_N[i, 0, j] - tildeyt1_N[i, 0, :])) * dt \
                                                - 1 / N * np.sum(grad_x2_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :],beta_t[i], beta2_t[i]) * (tildeyt1_N_v[i, 0, j] - tildeyt1_N_v[i, 0, :])) * dt
            elif kuramoto:
                for j in range(N):
                    tildeyt1_N[i + 1, 0, j] = tildeyt1_N[i, 0, j] \
                                              - grad_theta_grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 0, j] * dt \
                                              - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i]) * (tildeyt1_N[i, 0, j] - tildeyt1_N[1, 0, :])) * dt
            elif stochastic_volatility:
                tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] \
                                          - grad_theta_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * dt \
                                          - grad_x_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt \
                                          + 1.5 * sigma_t[i] * tildext1_N[i, :] ** 0.5 * tildeyt1_N[i, 0, :] * dwt1[i, :]

            else:
                if grad_w == grad_quadratic:
                    tildeyt1_N[i + 1, 0, :] = tildeyt1_N[i, 0, :] \
                                              - grad_theta_grad_v(tildext1_N[i, :], alpha_t[i]) * dt \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * tildeyt1_N[i, 0, :] * dt \
                                              - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 0, :] - np.mean(tildeyt1_N[i, 0, :])) * dt

                if grad_w != grad_quadratic:
                    for j in range(N):
                        tildeyt1_N[i + 1, 0, j] = tildeyt1_N[i, 0, j] \
                                                  - grad_theta_grad_v(tildext1_N[i, j], alpha_t[i]) * dt \
                                                  - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 0, j] * dt \
                                                  - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i]) * (tildeyt1_N[i, 0, j] - tildeyt1_N[1, 0, :])) * dt

        if est_alpha2:
            if stochastic_volatility:
                tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                          - grad_theta2_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * dt \
                                          - grad_x_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt \
                                          + 1.5 * sigma_t[i] * tildext1_N[i, :] ** 0.5 * tildeyt1_N[i, 1, :] * dwt1[i, :]
        if est_beta:
            if fitzhugh:
                tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                          - grad_x_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                          - grad_x2_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N_y[i, 1, :] * dt \
                                          - grad_theta_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt
                tildeyt1_N_y[i + 1, 1, :] = tildeyt1_N_y[i, 1, :] + tildeyt1_N[i + 1, 1, :] * dt
            elif cucker_smale:
                tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] + tildeyt1_N_v[i, 1, :] * dt
                for j in range(N):
                    tildeyt1_N_v[i + 1, 1, j] = tildeyt1_N_v[i, 1, j] \
                                                - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 1, j] * dt \
                                                - 1 / N * np.sum(grad_theta_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i])) * dt \
                                                - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]) * (tildeyt1_N[i, 1, j] - tildeyt1_N[i, 1, :])) * dt \
                                                - 1 / N * np.sum(grad_x2_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]) * (tildeyt1_N_v[i, 1, j] - tildeyt1_N_v[i, 1, :])) * dt
            elif kuramoto:
                for j in range(N):
                    tildeyt1_N[i + 1, 1, j] = tildeyt1_N[i, 1, j] \
                                              - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 1, j] * dt \
                                              - 1 / N * np.sum(grad_theta_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                              - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i]) * (tildeyt1_N[i, 1, j] - tildeyt1_N[1, 1, :])) * dt

            elif stochastic_volatility:
                tildeyt1_N[i + 1, 2, :] = tildeyt1_N[i, 2, :] \
                                          - grad_x_grad_v(tildext1_N[i, :], alpha_t[i], alpha2_t[i]) * tildeyt1_N[i, 2, :] * dt \
                                          - grad_theta_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 2, :] - np.mean(tildeyt1_N[i, 2, :])) * dt \
                                          + 1.5 * sigma_t[i] * tildext1_N[i, :] ** 0.5 * tildeyt1_N[i, 2, :] * dwt1[i, :]
            else:
                if grad_w == grad_quadratic:
                    tildeyt1_N[i + 1, 1, :] = tildeyt1_N[i, 1, :] \
                                              - grad_x_grad_v(tildext1_N[i, :], alpha_t[i]) * tildeyt1_N[i, 1, :] * dt \
                                              - grad_theta_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * dt \
                                              - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 1, :] - np.mean(tildeyt1_N[i, 1, :])) * dt
                if grad_w != grad_quadratic:
                    for j in range(N):
                        tildeyt1_N[i + 1, 1, j] = tildeyt1_N[i, 1, j] \
                                                  - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 1, j] * dt \
                                                  - 1 / N * np.sum(grad_theta_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i])) * dt \
                                                  - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], beta_t[i]) * (tildeyt1_N[i, 1, j] - tildeyt1_N[1, 1, :])) * dt

        if est_beta2:
            if cucker_smale:
                tildeyt1_N[i + 1, 2, :] = tildeyt1_N[i, 2, :] + tildeyt1_N_v[i, 2, :] * dt
                for j in range(N):
                    tildeyt1_N_v[i + 1, 2, j] = tildeyt1_N_v[i, 2, j] \
                                                - grad_x_grad_v(tildext1_N[i, j], alpha_t[i]) * tildeyt1_N[i, 2, j] * dt \
                                                - 1 / N * np.sum(grad_theta2_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i])) * dt \
                                                - 1 / N * np.sum(grad_x_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]) * (tildeyt1_N[i, 2, j] - tildeyt1_N[i, 2, :])) * dt \
                                                - 1 / N * np.sum(grad_x2_grad_w(tildext1_N[i, j] - tildext1_N[i, :], tildext1_N_v[i, j] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]) * (tildeyt1_N_v[i, 2, j] - tildeyt1_N_v[i, 2, :])) * dt

        if est_gamma:
            if fitzhugh:
                tildeyt1_N[i + 1, 2, :] = tildeyt1_N[i, 2, :] \
                                          - grad_x_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N[i, 2, :] * dt \
                                          - grad_x2_grad_v(tildext1_N[i, :], tildext1_N_y[i, :], alpha_t[i]) * tildeyt1_N_y[i, 2, :] * dt \
                                          - grad_x_grad_w(tildext1_N[i, :] - np.mean(tildext1_N[i, :]), beta_t[i]) * (tildeyt1_N[i, 2, :] - np.mean(tildeyt1_N[i, 2, :])) * dt
                tildeyt1_N_y[i + 1, 2, :] = tildeyt1_N_y[i, 2, :] + (1 + tildeyt1_N[i+1, 2, :]) * dt

        # online parameter updates
        # alpha_t
        if est_alpha:
            if fitzhugh:
                alpha_t[i + 1] = alpha_t[i] + all_gamma[i] \
                                 * (-grad_theta_grad_v(xt[i], yt[i], alpha_t[i])
                                    + averaging_func(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                 * (dxt - (- grad_v(xt[i], yt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            elif cucker_smale:
                alpha_t[i + 1] = alpha_t[i] + all_gamma[i] \
                                * (- grad_theta_grad_v(xt[i], alpha_t[i])
                                   - - averaging_func(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))
                                   - - averaging_func(tildeyt1_N_v[i, 0, :] * grad_x2_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))) \
                                * (dvt - (- grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], vt[i] - tildext2_N_v[i, :], beta_t[i], beta2_t[i]))) * dt)
            elif kuramoto:
                alpha_t[i + 1] = alpha_t[i] + all_gamma[i] \
                                 * (- grad_theta_grad_v(xt[i], alpha_t[i])
                                    + averaging_func(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                 * 1 / (sigma ** 2) \
                                 * (dxt - (-grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            elif stochastic_volatility:
                alpha_t[i + 1] = alpha_t[i] + all_gamma[i] \
                               * (- grad_theta_grad_v(xt[i], alpha_t[i], alpha2_t[i])
                                  + averaging_func(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                               * (dxt - (-grad_v(xt[i], alpha_t[i], alpha2_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            else:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] \
                               * (- grad_theta_grad_v(xt[i], alpha_t[i])
                                  + averaging_func(tildeyt1_N[i, 0, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                               * 1 / (sigma ** 2) \
                               * (dxt - (-grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)

        # alpha2_t (stochastic volatility only)
        if est_alpha2:
            if stochastic_volatility:
                alpha2_t[i + 1] = alpha2_t[i] + all_gamma[i] \
                               * (- grad_theta2_grad_v(xt[i], alpha_t[i], alpha2_t[i])
                                  + averaging_func(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                               * (dxt - (-grad_v(xt[i], alpha_t[i], alpha2_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)

        # beta_t
        if est_beta:
            if fitzhugh:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] \
                                * (- averaging_func(grad_theta_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))
                                   - - averaging_func(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                * (dxt - (- grad_v(xt[i], yt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            elif cucker_smale:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] \
                                * (- averaging_func(grad_theta_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_true[i], beta2_t[i]))
                                   - - averaging_func(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))
                                   - - averaging_func(tildeyt1_N_v[i, 1, :] * grad_x2_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))) \
                                * (dvt - (- grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], vt[i] - tildext2_N_v[i, :], beta_t[i], beta2_t[i]))) * dt)
            elif kuramoto:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] \
                                * (- averaging_func(grad_theta_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))
                                   - - averaging_func(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                * 1 / (sigma ** 2) * (dxt - (-grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            elif stochastic_volatility:
                beta_t[i + 1] = beta_t[i] + (50*all_gamma[i]) \
                                * (-averaging_func(grad_theta_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))
                                   + averaging_func(tildeyt1_N[i, 2, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                * (dxt - (-grad_v(xt[i], alpha_t[i], alpha2_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)
            else:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] \
                                * (-averaging_func(grad_theta_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))
                                   + averaging_func(tildeyt1_N[i, 1, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], beta_t[i]))) \
                                * 1/(sigma**2) * (dxt - (-grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt)

        # beta2_t (cucker-smale only)
        if est_beta2:
            if cucker_smale:
                beta2_t[i+1] = beta2_t[i] + all_gamma[i] \
                                * (-averaging_func(grad_theta2_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))
                                   - - averaging_func(tildeyt1_N[i, 2, :] * grad_x_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))
                                   - - averaging_func(tildeyt1_N_v[i, 2, :] * grad_x2_grad_w(xt[i] - tildext1_N[i, :], vt[i] - tildext1_N_v[i, :], beta_t[i], beta2_t[i]))) \
                                * (dvt - (- grad_v(xt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], vt[i] - tildext2_N_v[i, :], beta_t[i], beta2_t[i]))) * dt)

        # gamma_t (fitzhugh-nagumo only)
        if est_gamma:
            if fitzhugh:
                gamma_t[i + 1] = gamma_t[i] + all_gamma[i] \
                                 * (averaging_func(tildeyt1_N[i, 2, :] * grad_x_grad_w(xt[i]-tildext1_N[i,:], beta_t[i]))) \
                                 * (dxt - (- grad_v(xt[i], yt[i], alpha_t[i]) - averaging_func(grad_w(xt[i] - tildext2_N[i, :], beta_t[i]))) * dt) \
                                 + gamma \
                                 * (dyt - (gamma_t[i] + xt[i]) * dt)

        # sigma_t (stochastic volatility only)
        if est_sigma:
            if stochastic_volatility:
                sigma_t[i + 1] = sigma_t[i] + all_gamma[i] \
                                 * (2 * sigma_t[i] * xt[i] ** 3) \
                                 * (dxt ** 2 - sigma_t[i] ** 2 * xt[i] ** 3 * dt)

    if fitzhugh:
        return alpha_t, beta_t, gamma_t
    elif cucker_smale:
        return alpha_t, beta_t, beta2_t
    elif stochastic_volatility:
        return alpha_t, alpha2_t, beta_t, sigma_t
    else:
        return alpha_t, beta_t


#######################


##############################################
#### ONLINE ESTIMATORS (MVSDE LIKELIHOOD) ####
##############################################

## Recursive MLE

## Inputs:
## -> xtN (all observed paths of IPS)
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
## -> seed (random seed)
## -> whether to compute average over particles (estimator in SPA paper) or not (new estimator)

## Outputs:
## -> theta_t (online parameter estimate)

def online_est_ips(xtN, dt, grad_v, grad_theta_grad_v, alpha0, alpha_true,
                   est_alpha, grad_w, grad_theta_grad_w, beta0, beta_true, est_beta,
                   sigma, gamma, seed=1, average=True, fitzhugh=False, ytN=None, gamma0=None, gamma_true=None,
                   est_gamma=False, kuramoto=False, cucker_smale=False, vtN=None,
                   grad_theta2_grad_w=None, beta20=None, beta2_true=None, est_beta2=False,
                   stochastic_volatility=False, grad_theta2_grad_v=None, alpha20=None, alpha2_true=None,
                   est_alpha2=False, sigma0=None, sigma_true=None, est_sigma=False, opinion_dynamics=False):

    # check inputs
    if fitzhugh:
        assert ytN is not None
        assert gamma0 is not None
        assert gamma_true is not None
        assert grad_v == grad_fitzhugh
        assert grad_theta_grad_v == grad_theta_grad_fitzhugh
        assert grad_w == grad_quadratic
        assert grad_theta_grad_w == grad_theta_grad_quadratic

    if cucker_smale:
        assert vtN is not None
        assert beta20 is not None
        assert beta2_true is not None
        assert grad_v == grad_quadratic
        assert grad_theta_grad_v == grad_theta_grad_quadratic
        assert grad_w == grad_cucker_smale
        assert grad_theta_grad_w == grad_theta1_grad_cucker_smale
        assert grad_theta2_grad_w == grad_theta2_grad_cucker_smale

    if kuramoto:
        assert grad_w == grad_kuramoto
        assert grad_theta_grad_w == grad_theta_grad_kuramoto

    if stochastic_volatility:
        assert alpha20 is not None
        assert alpha2_true is not None
        assert sigma0 is not None
        assert sigma_true is not None
        assert grad_v == grad_stochastic_volatility
        assert grad_theta_grad_v == grad_theta1_grad_stochastic_volatility
        assert grad_theta2_grad_v == grad_theta2_grad_stochastic_volatility
        assert grad_w == grad_quadratic
        assert grad_theta_grad_w == grad_theta_grad_quadratic

    if opinion_dynamics:
        assert beta20 is not None
        assert beta2_true is not None
        assert grad_v == grad_linear
        assert grad_theta_grad_v == grad_theta_grad_linear
        assert grad_w == grad_opinion_dynamics
        assert grad_theta_grad_w == grad_theta1_grad_opinion_dynamics
        assert grad_theta2_grad_w == grad_theta2_grad_opinion_dynamics

    # averaging func
    if average:
        def averaging_func1(x):
            return np.mean(x)
        def averaging_func2(x):
            return np.mean(x)
    if not average:
        def averaging_func1(x):
            return x[1]
        def averaging_func2(x):
            return x[2]

    # set random seed
    np.random.seed(seed)

    # number of time steps
    nt = xt.shape[0] - 1

    # learning rate
    if type(gamma) is float or type(gamma) is int:
        all_gamma = [gamma] * (nt + 1)
    else:
        all_gamma = gamma

    # parameters
    if type(alpha_true) is float or type(alpha_true) is int:
        alpha_true = [alpha_true] * (nt + 1)

    if type(beta_true) is float or type(beta_true) is int:
        beta_true = [beta_true] * (nt + 1)

    if fitzhugh:
        if type(gamma_true) is float or type(gamma_true) is int:
            gamma_true = [gamma_true] * (nt + 1)

    if cucker_smale or opinion_dynamics:
        if type(beta2_true) is float or type(beta2_true) is int:
            beta2_true = [beta2_true] * (nt + 1)

    if stochastic_volatility:
        if type(alpha2_true) is float or type(alpha2_true) is int:
            alpha2_true = [alpha2_true] * (nt + 1)

        if type(sigma_true) is float or type(sigma_true) is int:
            sigma_true = [sigma_true] * (nt + 1)

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

    gamma_t = np.zeros(nt + 1)
    if fitzhugh:
        if est_gamma:
            gamma_t[0] = gamma0
        else:
            gamma_t = gamma_true

    beta2_t = np.zeros(nt + 1)
    if cucker_smale or opinion_dynamics:
        if est_beta2:
            beta2_t[0] = beta20
        else:
            beta2_t = beta2_true

    alpha2_t = np.zeros(nt + 1)
    sigma_t = np.zeros(nt + 1)
    if stochastic_volatility:
        if est_alpha2:
            alpha2_t[0] = alpha20
        else:
            alpha2_t = alpha2_true

        if est_sigma:
            sigma_t[0] = sigma0
        else:
            sigma_t = sigma_true

    # integrate parameter update equations
    for i in tqdm(range(0, nt)):

        t = i * dt
        dxt = xtN[i + 1, :] - xtN[i, :]

        if fitzhugh:
            dyt = ytN[i+1, :] - ytN[i, :]

        if cucker_smale:
            dvt = vtN[i+1, :] - vtN[i, :]

        # online parameter updates
        # alpha_t
        if est_alpha:
            if fitzhugh:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] * (-grad_theta_grad_v(xtN[i, 0], ytN[i, 0], alpha_t[i])) * (dxt[0] - (- grad_v(xtN[i, 0], ytN[i, 0], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif cucker_smale:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] * (- grad_theta_grad_v(xtN[i, 0], alpha_t[i])) * (dvt[0] - (- grad_v(xtN[i,0], alpha_t[i]) - averaging_func2(grad_w(xtN[i,0] - xtN[i, :], vtN[i, 0] - vtN[i, :], beta_t[i], beta2_t[i]))) * dt)
            elif kuramoto:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] * (- grad_theta_grad_v(xtN[i, 0], alpha_t[i])) * 1 / (sigma ** 2) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif stochastic_volatility:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] * (- grad_theta_grad_v(xtN[i, 0], alpha_t[i], alpha2_t[i])) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i], alpha2_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif opinion_dynamics:
                alpha_t[i + 1] = alpha_t[i] + all_gamma[i] * (-grad_theta_grad_v(xtN[i, 0], alpha_t[i])) * 1 / (sigma ** 2) * (dxt[0] - (- grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(np.vectorize(grad_w, otypes=[np.float64])(xtN[i, 0] - xtN[i, :], beta_t[i], beta2_t[i]))) * dt)
            else:
                alpha_t[i+1] = alpha_t[i] + all_gamma[i] * (-grad_theta_grad_v(xtN[i, 0], alpha_t[i])) * 1 / (sigma ** 2) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)

        # alpha2_t (stochastic volatility only)
        if est_alpha2:
            if stochastic_volatility:
                alpha2_t[i+1] = alpha2_t[i] + all_gamma[i] * (- grad_theta2_grad_v(xtN[i, 0], alpha_t[i], alpha2_t[i])) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i], alpha2_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)

        # beta_t
        if est_beta:
            if fitzhugh:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] * (- averaging_func1(grad_theta_grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * (dxt[0] - (- grad_v(xtN[i, 0], ytN[i, 0], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif cucker_smale:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] * (- averaging_func1(grad_theta_grad_w(xtN[i, 0] - xtN[i, :], vtN[i, 0] - vtN[i, :], beta_t[i], beta2_t[i]))) * (dvt[0] - (- grad_v(xtN[i,0], alpha_t[i]) - averaging_func2(grad_w(xtN[i,0] - xtN[i, :], vtN[i, 0] - vtN[i, :], beta_t[i], beta2_t[i]))) * dt)
            elif kuramoto:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] * (- averaging_func1(grad_theta_grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * 1 / (sigma ** 2) * (dxt[0] - (-grad_v(xtN[i], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif stochastic_volatility:
                beta_t[i + 1] = beta_t[i] + 100 * (all_gamma[i]) * (-averaging_func1(grad_theta_grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i], alpha2_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)
            elif opinion_dynamics:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] * (-averaging_func1(np.vectorize(grad_theta_grad_w, otypes=[np.float64])(xtN[i, 0] - xtN[i, :], beta_t[i], beta2_t[i]))) * 1 / (sigma ** 2) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(np.vectorize(grad_w, otypes=[np.float64])(xtN[i, 0] - xtN[i, :], beta_t[i], beta2_t[i]))) * dt)
            else:
                beta_t[i + 1] = beta_t[i] + all_gamma[i] * (-averaging_func1(grad_theta_grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * 1 / (sigma ** 2) * (dxt[0] - (-grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(grad_w(xtN[i, 0] - xtN[i, :], beta_t[i]))) * dt)

        # beta2_t (cucker-smale only)
        if est_beta2:
            if cucker_smale:
                beta2_t[i + 1] = beta2_t[i] + all_gamma[i] * (-averaging_func1(grad_theta2_grad_w(xtN[i, 0] - xtN[i, :], vtN[i, 0] - vtN[i, :], beta_t[i], beta2_t[i]))) * (dvt[0] - (- grad_v(xtN[i,0], alpha_t[i]) - averaging_func2(grad_w(xtN[i,0] - xtN[i, :], vtN[i, 0] - vtN[i, :], beta_t[i], beta2_t[i]))) * dt)
            if opinion_dynamics:
                update = 0
                for j in range(N_par):
                    update += (-averaging_func1(np.vectorize(grad_theta2_grad_w, otypes=[np.float64])(xtN[i, j] - xtN[i, :], beta_t[i], beta2_t[i]))) * 1 / (sigma ** 2) * (dxt[j] - (- grad_v(xtN[i, j], alpha_t[i]) - averaging_func2(np.vectorize(grad_w, otypes=[np.float64])(xtN[i, j] - xtN[i, :], beta_t[i], beta2_t[i]))) * dt)
                beta2_t[i + 1] = beta2_t[i] + all_gamma[i] * 1 / N_par * update
                #beta2_t[i + 1] = beta2_t[i] + all_gamma[i] * (-averaging_func1(np.vectorize(grad_theta2_grad_w, otypes=[np.float64])(xtN[i, 0] - xtN[i, :], beta_t[i], beta2_t[i]))) * 1 / (sigma ** 2) * (dxt[0] - (- grad_v(xtN[i, 0], alpha_t[i]) - averaging_func2(np.vectorize(grad_w, otypes=[np.float64])(xtN[i, 0] - xtN[i, :], beta_t[i], beta2_t[i]))) * dt)

        # gamma_t (fitzhugh-nagumo only)
        if est_gamma:
            if fitzhugh:
                gamma_t[i + 1] = gamma_t[i] + all_gamma[i] * (dyt[0] - (gamma_t[i] + xtN[i, 0]) * dt)

        # sigma_t (stochastic volatility only)
        if est_sigma:
            if stochastic_volatility:
                sigma_t[i + 1] = sigma_t[i] + 10 * all_gamma[i] * (2 * sigma_t[i] * xtN[i, 0] ** 3) * (dxt[0] ** 2 - sigma_t[i] ** 2 * xtN[i, 0] ** 3 * dt)

    if fitzhugh:
        return alpha_t, beta_t, gamma_t
    elif cucker_smale or opinion_dynamics:
        return alpha_t, beta_t, beta2_t
    elif stochastic_volatility:
        return alpha_t, alpha2_t, beta_t, sigma_t
    else:
        return alpha_t, beta_t


#######################


if __name__ == "__main__":

    # general
    root = "results/"
    leaf = "opinion_dynamics"
    path = os.path.join(root, leaf)
    if not os.path.exists(path):
        os.makedirs(path)

    # output
    save_plots = False

    # simulation parameters
    N_par = 20
    T = 500
    dt = 0.1

    quadratic = False
    bistable = False
    kuramoto = False
    fitzhugh = False
    cucker_smale = False
    stochastic_volatility = False
    opinion_dynamics = False

    if quadratic:
        grad_v = grad_quadratic
        grad_theta_grad_v = grad_theta_grad_quadratic
        grad_x_grad_v = grad_x_grad_quadratic
        grad_w = grad_quadratic
        grad_theta_grad_w = grad_theta_grad_quadratic
        grad_x_grad_w = grad_x_grad_quadratic

    if bistable:
        grad_v = grad_bi_stable
        grad_theta_grad_v = grad_theta_grad_bi_stable
        grad_x_grad_v = grad_x_grad_bi_stable
        grad_w = grad_quadratic
        grad_theta_grad_w = grad_theta_grad_quadratic
        grad_x_grad_w = grad_x_grad_quadratic

    if fitzhugh:
        grad_v = grad_fitzhugh
        grad_theta_grad_v = grad_theta_grad_fitzhugh
        grad_x_grad_v = grad_x1_grad_fitzhugh
        grad_x2_grad_v = grad_x2_grad_fitzhugh
        grad_w = grad_quadratic
        grad_theta_grad_w = grad_theta_grad_quadratic
        grad_x_grad_w = grad_x_grad_quadratic

    if cucker_smale:
        grad_v = grad_quadratic
        grad_theta_grad_v = grad_theta_grad_quadratic
        grad_x_grad_v = grad_x_grad_quadratic
        grad_w = grad_cucker_smale
        grad_theta_grad_w = grad_theta1_grad_cucker_smale
        grad_theta2_grad_w = grad_theta2_grad_cucker_smale
        grad_x_grad_w = grad_x1_grad_cucker_smale
        grad_x2_grad_w = grad_x2_grad_cucker_smale

    if kuramoto:
        grad_v = grad_quadratic
        grad_theta_grad_v = grad_theta_grad_quadratic
        grad_x_grad_v = grad_x_grad_quadratic
        grad_w = grad_kuramoto
        grad_theta_grad_w = grad_theta_grad_kuramoto
        grad_x_grad_w = grad_x_grad_kuramoto

    if stochastic_volatility:
        grad_v = grad_stochastic_volatility
        grad_theta_grad_v = grad_theta1_grad_stochastic_volatility
        grad_theta2_grad_v = grad_theta2_grad_stochastic_volatility
        grad_x_grad_v = grad_x_grad_stochastic_volatility
        grad_w = grad_quadratic
        grad_theta_grad_w = grad_theta_grad_quadratic
        grad_x_grad_w = grad_x_grad_quadratic

    if opinion_dynamics:
        grad_v = grad_linear
        grad_theta_grad_v = grad_theta_grad_linear
        grad_w = grad_opinion_dynamics
        grad_theta_grad_w = grad_theta1_grad_opinion_dynamics
        grad_theta2_grad_w = grad_theta2_grad_opinion_dynamics

    seeds = range(1)

    nt = round(T / dt)
    t = [i * dt for i in range(nt + 1)]

    # step size
    gamma = .01 #[min(.1, .1/(1+t_val)**(.5)) for t_val in t] #0.005

    # parameters
    alpha0 = 3.0
    alpha_true = 3.5
    est_alpha = False

    alpha20 = 3.5
    alpha2_true = 3.5
    est_alpha2 = False

    beta0 = 0.2
    beta_true = 2.
    est_beta = False

    beta20 = 1.5
    beta2_true = 1.0
    est_beta2 = False

    gamma0 = 1.0
    gamma_true = 0.3
    est_gamma = False

    sigma0 = 0.5
    sigma_true = .1
    est_sigma = False

    N_est = 50

    # plotting
    plot_each_run = False
    plot_mean_run = True

    all_alpha_t = np.zeros((nt + 1, len(seeds), 2))
    all_beta_t = np.zeros((nt + 1, len(seeds), 2))
    all_alpha_ips_t = np.zeros((nt + 1, len(seeds), 2))
    all_beta_ips_t = np.zeros((nt + 1, len(seeds), 2))

    if fitzhugh:
        all_gamma_t = np.zeros((nt + 1, len(seeds), 2))
        all_gamma_ips_t = np.zeros((nt + 1, len(seeds), 2))

    if cucker_smale or opinion_dynamics:
        all_beta2_t = np.zeros((nt + 1, len(seeds), 2))
        all_beta2_ips_t = np.zeros((nt + 1, len(seeds), 2))

    if stochastic_volatility:
        all_alpha2_t = np.zeros((nt + 1, len(seeds), 2))
        all_sigma_t = np.zeros((nt + 1, len(seeds), 2))
        all_alpha2_ips_t = np.zeros((nt + 1, len(seeds), 2))
        all_sigma_ips_t = np.zeros((nt + 1, len(seeds), 2))

    for idx, seed in enumerate(seeds):
        print(seed)

        # simulate mvsde
        x0 = np.random.normal(0, 1, N_par) #(-1,1,N_par) #np.random.uniform(0, 3, N_par) #np.#np.random.uniform(-1, 1, N_par)
        y0 = np.random.normal(0, 1, N_par)
        v0 = np.random.uniform(-1, 1, N_par) #np.random.normal(0, 1, N_par)

        if fitzhugh:
            xtN, ytN = sde_sim_func(N=N_par, T=T, grad_v=grad_v, alpha=alpha_true, grad_w=grad_w, beta=beta_true,
                                  Aij=None, Lij=None, sigma=sigma_true, x0=x0, dt=dt, seed=seed, kuramoto=kuramoto,
                                  fitzhugh=fitzhugh, y0=y0, gamma=gamma_true, cucker_smale=cucker_smale, v0=v0,
                                  beta2=beta2_true, stochastic_volatility=stochastic_volatility, alpha2=alpha2_true,
                                  all_particles=True)
            xt, yt = xtN[:, 0], ytN[:, 0]
        elif cucker_smale:
            xtN, vtN = sde_sim_func(N=N_par, T=T, grad_v=grad_v, alpha=alpha_true, grad_w=grad_w, beta=beta_true,
                                  Aij=None, Lij=None, sigma=sigma_true, x0=x0, dt=dt, seed=seed, kuramoto=kuramoto,
                                  fitzhugh=fitzhugh, y0=y0, gamma=gamma_true, cucker_smale=cucker_smale, v0=v0,
                                  beta2=beta2_true, stochastic_volatility=stochastic_volatility, alpha2=alpha2_true,
                                  all_particles=True)
            xt, vt = xtN[:, 0], vtN[:, 0]
        else:
            xtN = sde_sim_func(N=N_par, T=T, grad_v=grad_v, alpha=alpha_true, grad_w=grad_w, beta=beta_true,
                                  Aij=None, Lij=None, sigma=sigma_true, x0=x0, dt=dt, seed=seed, kuramoto=kuramoto,
                                  fitzhugh=fitzhugh, y0=y0, gamma=gamma_true, cucker_smale=cucker_smale, v0=v0,
                                  beta2=beta2_true, stochastic_volatility=stochastic_volatility, alpha2=alpha2_true,
                                  all_particles=True, opinion_dynamics=opinion_dynamics)
            xt = xtN[:, 0]

        # parameter estimation
        if True:
            if fitzhugh:
                #alpha_t_one, beta_t_one, gamma_t_one = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v,
                #                                                  alpha0, alpha_true, est_alpha, grad_w,
                #                                                  grad_theta_grad_w, grad_x_grad_w, beta0, beta_true,
                #                                                  est_beta, sigma_true, gamma, N_est, seed,
                #                                                  average=True,  fitzhugh=fitzhugh, yt=yt,
                #                                                  grad_x2_grad_v=grad_x2_grad_v, gamma0=gamma0,
                #                                                  gamma_true=gamma_true, est_gamma=est_gamma)
                #alpha_t_two, beta_t_two, gamma_t_two = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v,
                #                                                  alpha0, alpha_true, est_alpha, grad_w,
                #                                                  grad_theta_grad_w, grad_x_grad_w, beta0, beta_true,
                #                                                  est_beta, sigma_true, gamma, N_est, seed,
                #                                                  average=False,  fitzhugh=fitzhugh, yt=yt,
                #                                                  grad_x2_grad_v=grad_x2_grad_v, gamma0=gamma0,
                #                                                  gamma_true=gamma_true, est_gamma=est_gamma)
                alpha_t_ips_one, beta_t_ips_one, gamma_t_ips_one = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha,
                                                                                  grad_w, grad_theta_grad_w,
                                                                                  beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma,
                                                                                  seed, average=True, fitzhugh=fitzhugh,
                                                                                  ytN=ytN, gamma0=gamma0,
                                                                                  gamma_true=gamma_true,
                                                                                  est_gamma=est_gamma)
                alpha_t_ips_two, beta_t_ips_two, gamma_t_ips_two = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha, grad_w,
                                                                                  grad_theta_grad_w, beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma, seed,
                                                                                  average=False, fitzhugh=fitzhugh,
                                                                                  ytN=ytN, gamma0=gamma0,
                                                                                  gamma_true=gamma_true,
                                                                                  est_gamma=est_gamma)
            elif cucker_smale:
                #alpha_t_one, beta_t_one, beta2_t_one = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v,
                #                                                  alpha0, alpha_true, est_alpha, grad_w,
                #                                                  grad_theta_grad_w, grad_x_grad_w, beta0, beta_true,
                #                                                  est_beta, sigma_true, gamma, N_est, seed,
                #                                                  cucker_smale=True, vt=vt,
                #                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                #                                                  grad_x2_grad_w=grad_x2_grad_w, beta20=beta20,
                #                                                  beta2_true=beta2_true, est_beta2=est_beta2,
                #                                                  average=True)
                #alpha_t_two, beta_t_two, beta2_t_two = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v,
                #                                                  alpha0, alpha_true, est_alpha, grad_w,
                #                                                  grad_theta_grad_w, grad_x_grad_w, beta0, beta_true,
                #                                                  est_beta, sigma_true, gamma, N_est, seed,
                #                                                  cucker_smale=True, vt=vt,
                #                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                #                                                  grad_x2_grad_w=grad_x2_grad_w, beta20=beta20,
                #                                                  beta2_true=beta2_true, est_beta2=est_beta2,
                #                                                  average=False)
                alpha_t_ips_one, beta_t_ips_one, beta2_t_ips_one = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha, grad_w,
                                                                                  grad_theta_grad_w, beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma,
                                                                                  seed, cucker_smale=True, vtN=vtN,
                                                                                  beta20=beta20, beta2_true=beta2_true,
                                                                                  est_beta2=est_beta2,
                                                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                                                                                  average=True)
                alpha_t_ips_two, beta_t_ips_two, beta2_t_ips_two = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha, grad_w,
                                                                                  grad_theta_grad_w, beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma,
                                                                                  seed, cucker_smale=True, vtN=vtN,
                                                                                  beta20=beta20, beta2_true=beta2_true,
                                                                                  est_beta2=est_beta2,
                                                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                                                                                  average=False)
            elif opinion_dynamics:
                alpha_t_ips_one, beta_t_ips_one, beta2_t_ips_one = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha, grad_w,
                                                                                  grad_theta_grad_w, beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma,
                                                                                  seed, opinion_dynamics=True,
                                                                                  beta20=beta20, beta2_true=beta2_true,
                                                                                  est_beta2=est_beta2,
                                                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                                                                                  average=True)
                alpha_t_ips_two, beta_t_ips_two, beta2_t_ips_two = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v,
                                                                                  alpha0, alpha_true, est_alpha, grad_w,
                                                                                  grad_theta_grad_w, beta0, beta_true,
                                                                                  est_beta, sigma_true, gamma,
                                                                                  seed, opinion_dynamics=True,
                                                                                  beta20=beta20, beta2_true=beta2_true,
                                                                                  est_beta2=est_beta2,
                                                                                  grad_theta2_grad_w=grad_theta2_grad_w,
                                                                                  average=False)
            elif stochastic_volatility:
                #alpha_t_one, alpha2_t_one, beta_t_one, sigma_t_one = online_est(xt, dt, grad_v, grad_theta_grad_v,
                #                                                                alpha0, alpha_true,
                #                                                                est_alpha, grad_w, grad_theta_grad_w,
                #                                                                grad_x_grad_w, beta0, beta_true,
                #                                                                est_beta, sigma_true, gamma, N_est,
                #                                                                seed, stochastic_volatility=True,
                #                                                                grad_theta2_grad_v=grad_theta2_grad_v,
                #                                                                alpha20=alpha20, alpha2_true=alpha2_true,
                #                                                                est_alpha2=est_alpha2, sigma0=sigma0,
                #                                                                sigma_true=sigma_true,
                #                                                                est_sigma=est_sigma, average=True)
                #alpha_t_two, alpha2_t_two, beta_t_two, sigma_t_two = online_est(xt, dt, grad_v, grad_theta_grad_v,
                #                                                                grad_x_grad_v, alpha0, alpha_true,
                #                                                                est_alpha, grad_w, grad_theta_grad_w,
                #                                                                grad_x_grad_w, beta0, beta_true,
                #                                                                est_beta, sigma_true, gamma, N_est,
                #                                                                seed, stochastic_volatility=True,
                #                                                                grad_theta2_grad_v=grad_theta2_grad_v,
                #                                                                alpha20=alpha20,
                #                                                                alpha2_true=alpha2_true,
                #                                                                est_alpha2=est_alpha2, sigma0=sigma0,
                #                                                                sigma_true=sigma_true,
                #                                                                est_sigma=est_sigma, average=False)
                alpha_t_ips_one, alpha2_t_ips_one, beta_t_ips_one, sigma_t_ips_one = online_est_ips(xtN, dt, grad_v,
                                                                                                    grad_theta_grad_v,
                                                                                                    alpha0, alpha_true,
                                                                                                    est_alpha, grad_w,
                                                                                                    grad_theta_grad_w,
                                                                                                    beta0, beta_true,
                                                                                                    est_beta, sigma_true,
                                                                                                    gamma, seed,
                                                                                                    stochastic_volatility=True,
                                                                                                    grad_theta2_grad_v=grad_theta2_grad_v,
                                                                                                    alpha20=alpha20,
                                                                                                    alpha2_true=alpha2_true,
                                                                                                    est_alpha2=est_alpha2,
                                                                                                    sigma0=sigma0,
                                                                                                    sigma_true=sigma_true,
                                                                                                    est_sigma=est_sigma,
                                                                                                    average=True)
                alpha_t_ips_two, alpha2_t_ips_two, beta_t_ips_two, sigma_t_ips_two = online_est_ips(xtN, dt, grad_v,
                                                                                                    grad_theta_grad_v,
                                                                                                    alpha0, alpha_true,
                                                                                                    est_alpha, grad_w,
                                                                                                    grad_theta_grad_w,
                                                                                                    beta0, beta_true,
                                                                                                    est_beta, sigma_true,
                                                                                                    gamma, seed,
                                                                                                    stochastic_volatility=True,
                                                                                                    grad_theta2_grad_v=grad_theta2_grad_v,
                                                                                                    alpha20=alpha20,
                                                                                                    alpha2_true=alpha2_true,
                                                                                                    est_alpha2=est_alpha2,
                                                                                                    sigma0=sigma0,
                                                                                                    sigma_true=sigma_true,
                                                                                                    est_sigma=est_sigma,
                                                                                                    average=False)
            else:
                #alpha_t_one, beta_t_one = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true,
                #                                     est_alpha, grad_w, grad_theta_grad_w, grad_x_grad_w, beta0,
                #                                     beta_true, est_beta, sigma_true, gamma, N_est, seed, kuramoto=kuramoto,
                #                                     average=True)
                #alpha_t_two, beta_t_two = online_est(xt, dt, grad_v, grad_theta_grad_v, grad_x_grad_v, alpha0, alpha_true,
                #                                     est_alpha, grad_w, grad_theta_grad_w, grad_x_grad_w, beta0,
                #                                     beta_true, est_beta, sigma_true, gamma, N_est, seed, kuramoto=kuramoto,
                #                                     average=False)
                alpha_t_ips_one, beta_t_ips_one = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v, alpha0,
                                                                 alpha_true, est_alpha, grad_w, grad_theta_grad_w,
                                                                 beta0, beta_true, est_beta, sigma_true, gamma, seed,
                                                                 kuramoto=kuramoto, average=True)
                alpha_t_ips_two, beta_t_ips_two = online_est_ips(xtN, dt, grad_v, grad_theta_grad_v, alpha0,
                                                                 alpha_true, est_alpha, grad_w, grad_theta_grad_w,
                                                                 beta0, beta_true, est_beta, sigma_true,
                                                                 gamma, seed, kuramoto=kuramoto, average=False)


            if fitzhugh:
                #all_alpha_t[:, idx, 0], all_beta_t[:, idx, 0], all_gamma_t[:, idx, 0] = alpha_t_one, beta_t_one, gamma_t_one
                #all_alpha_t[:, idx, 1], all_beta_t[:, idx, 1], all_gamma_t[:, idx, 1] = alpha_t_two, beta_t_two, gamma_t_two
                all_alpha_ips_t[:, idx, 0], all_beta_ips_t[:, idx, 0], all_gamma_ips_t[:, idx, 0] = alpha_t_ips_one, beta_t_ips_one, gamma_t_ips_one
                all_alpha_ips_t[:, idx, 1], all_beta_ips_t[:, idx, 1], all_gamma_ips_t[:, idx, 1] = alpha_t_ips_two, beta_t_ips_two, gamma_t_ips_two
            elif cucker_smale or opinion_dynamics:
                #all_alpha_t[:, idx, 0], all_beta_t[:, idx, 0], all_beta2_t[:, idx, 0] = alpha_t_one,beta_t_one, beta2_t_one
                #all_alpha_t[:, idx, 1], all_beta_t[:, idx, 1], all_beta2_t[:, idx, 1] = alpha_t_two, beta_t_two, beta2_t_two
                all_alpha_ips_t[:, idx, 0], all_beta_ips_t[:, idx, 0], all_beta2_ips_t[:, idx, 0] = alpha_t_ips_one, beta_t_ips_one, beta2_t_ips_one
                all_alpha_ips_t[:, idx, 1], all_beta_ips_t[:, idx, 1], all_beta2_ips_t[:, idx, 1] = alpha_t_ips_two, beta_t_ips_two, beta2_t_ips_two
            elif stochastic_volatility:
                #all_alpha_t[:, idx, 0], all_alpha2_t[:, idx, 0], all_beta_t[:, idx, 0], all_sigma_t[:, idx, 0] = alpha_t_one, alpha2_t_one, beta_t_one, sigma_t_one
                #all_alpha_t[:, idx, 1], all_alpha2_t[:, idx, 1], all_beta_t[:, idx, 1], all_sigma_t[:, idx, 1] = alpha_t_two, alpha2_t_two, beta_t_two, sigma_t_two
                all_alpha_ips_t[:, idx, 0], all_alpha2_ips_t[:, idx, 0], all_beta_ips_t[:, idx, 0], all_sigma_ips_t[:, idx, 0] = alpha_t_ips_one, alpha2_t_ips_one, beta_t_ips_one, sigma_t_ips_one
                all_alpha_ips_t[:, idx, 1], all_alpha2_ips_t[:, idx, 1], all_beta_ips_t[:, idx, 1], all_sigma_ips_t[:, idx, 1] = alpha_t_ips_two, alpha2_t_ips_two, beta_t_ips_two, sigma_t_ips_two
            else:
                #all_alpha_t[:, idx, 0], all_beta_t[:, idx, 0] = alpha_t_one, beta_t_one
                #all_alpha_t[:, idx, 1], all_beta_t[:, idx, 1] = alpha_t_two, beta_t_two
                all_alpha_ips_t[:, idx, 0], all_beta_ips_t[:, idx, 0] = alpha_t_ips_one, beta_t_ips_one
                all_alpha_ips_t[:, idx, 1], all_beta_ips_t[:, idx, 1] = alpha_t_ips_two, beta_t_ips_two

            if plot_each_run:
                if est_alpha and not est_alpha2 and not est_beta and not est_beta2 and not est_gamma:
                    #plt.plot(t, alpha_t_one, label=r"$\alpha_{t}^N$ (Estimator 1)", color="C0")
                    #plt.plot(t, alpha_t_two, label=r"$\alpha_{t}^N$ (Estimator 2)", color="C1")
                    plt.plot(t, alpha_t_ips_one, label=r"$\alpha_{t}^N$ (IPS Estimator 1)", color="C0")
                    plt.plot(t, alpha_t_ips_two, label=r"$\alpha_{t}^N$ (IPS Estimator 2)", color="C1")
                    plt.axhline(y=alpha_true, linestyle="--", color="black")
                    plt.legend()
                    plt.show()
                    plt.pause(2)
                    plt.close()
                if est_beta and not est_alpha and not est_alpha2 and not est_beta2 and not est_gamma:
                    #plt.plot(t, beta_t_one, label=r"$\beta_{t}^N$ (Estimator 1)", color="C0")
                    #plt.plot(t, beta_t_two, label=r"$\beta_{t}^N$ (Estimator 2)", color="C1")
                    plt.plot(t, beta_t_ips_one, label=r"$\beta_{t}^N$ (IPS Estimator 1)", color="C0")
                    plt.plot(t, beta_t_ips_two, label=r"$\beta_{t}^N$ (IPS Estimator 2)", color="C1")
                    plt.axhline(y=beta_true, linestyle="--", color="black")
                    plt.legend()
                    plt.show()
                    plt.pause(2)
                    plt.close()
                if est_gamma and not est_alpha and not est_alpha2 and not est_beta and not est_beta2:
                    #plt.plot(t, gamma_t_one, label=r"$\gamma_{t}^N$ (Estimator 1)", color="C0")
                    #plt.plot(t, gamma_t_two, label=r"$\gamma_{t}^N$ (Estimator 2)", color="C1")
                    plt.plot(t, gamma_t_ips_one, label=r"$\gamma_{t}^N$ (IPS Estimator 1)", color="C0")
                    plt.plot(t, gamma_t_ips_two, label=r"$\gamma_{t}^N$ (IPS Estimator 2)", color="C1")
                    plt.axhline(y=gamma_true, linestyle="--", color="black")
                    plt.legend()
                    plt.show()
                    plt.pause(2)
                    plt.close()
                if est_beta2 and not est_alpha and not est_alpha2 and not est_beta and not est_gamma:
                    # plt.plot(t, beta2_t_one, label=r"$\beta_{2,t}^N$ (Estimator 1)", color="C0")
                    # plt.plot(t, beta2_t_two, label=r"$\beta_{2,t}^N$ (Estimator 2)", color="C1")
                    plt.plot(t, beta2_t_ips_one, label=r"$\beta_{2,t}^N$ (IPS Estimator 1)", color="C0")
                    plt.plot(t, beta2_t_ips_two, label=r"$\beta_{2,t}^N$ (IPS Estimator 2)", color="C1")
                    plt.axhline(y=beta2_true, linestyle="--", color="black")
                    plt.legend()
                    plt.show()
                    plt.pause(2)
                    plt.close()
    if True:
        if plot_mean_run:
            if plot_each_run:
                cols = ["C1", "C2"]
            else:
                cols = ["C0", "C1"]
            if est_alpha and not est_alpha2 and not est_beta and not est_gamma and not est_sigma:
                #plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=alpha_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_est_all_ips.eps", dpi=300)
                plt.show()
            if est_alpha2 and not est_alpha and not est_beta and not est_gamma and not est_sigma:
                plt.plot(t, np.mean(all_alpha2_t[:, :, 0], 1), label=r"$\theta_{t,2}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_alpha2_t[:, :, 1], 1), label=r"$\theta_{t,2}^N$ (Estimator 2)")
                plt.axhline(y=alpha2_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha2_est_all.eps", dpi=300)
                plt.show()
            elif est_beta and not est_alpha and not est_alpha2 and not est_gamma and not est_sigma:
                #plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/beta_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_gamma and not est_alpha and not est_alpha2 and not est_beta and not est_sigma:
                #plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=gamma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/gamma_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_beta2 and not est_alpha and not est_alpha2 and not est_beta and not est_sigma:
                #plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\beta_{2,t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\beta_{2,t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta2_ips_t[:, :, 0], 1), label=r"$\beta_{2,t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta2_ips_t[:, :, 1], 1), label=r"$\beta_{2,t}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta2_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/beta2_est_all_ips_ex2.eps", dpi=300)
                plt.show()
            elif est_sigma and not est_alpha and not est_alpha2 and not est_beta and not est_gamma:
                plt.plot(t, np.mean(all_sigma_t[:, :, 0], 1), label=r"$\eta_{t,1}^N$ (Estimator 1)")
                plt.axhline(y=sigma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/sigma_est_all.eps", dpi=300)
                plt.show()
            elif est_alpha and est_beta and not est_alpha2 and not est_gamma and not est_sigma:
                #plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=alpha_true, linestyle="--", color="black")
                #plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_beta_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_alpha and est_alpha2 and not est_beta and not est_gamma and not est_sigma:
                plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\theta_{t,1}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\theta_{t,1}^N$ (Estimator 2)")
                plt.axhline(y=alpha_true, linestyle="--", color="black")
                plt.plot(t, np.mean(all_alpha2_t[:, :, 0], 1), label=r"$\theta_{t,2}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_alpha2_t[:, :, 1], 1), label=r"$\theta_{t,2}^N$ (Estimator 2)")
                plt.axhline(y=alpha2_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_alpha2_est_all.eps", dpi=300)
                plt.show()
            elif est_alpha and est_gamma and not est_beta and not est_alpha2 and not est_sigma:
                # plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=alpha_true, linestyle="--", color="black")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=gamma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_gamma_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_alpha and est_sigma and not est_beta and not est_alpha2 and not est_gamma:
                plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\theta_{t,3}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\theta_{t,3}^N$ (Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                plt.plot(t, np.mean(all_sigma_t[:, :, 0], 1), label=r"$\eta_{t,1}^N$")
                plt.axhline(y=sigma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_sigma_est_all.eps", dpi=300)
                plt.show()
            elif est_alpha2 and est_beta and not est_alpha and not est_gamma and not est_sigma:
                plt.plot(t, np.mean(all_alpha2_t[:, :, 0], 1), label=r"$\theta_{t,2}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_alpha2_t[:, :, 1], 1), label=r"$\theta_{t,2}^N$ (Estimator 2)")
                plt.axhline(y=alpha2_true, linestyle="--", color="black")
                plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\theta_{t,3}^N$ (Estimator 1)")
                plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\theta_{t,3}^N$ (Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha2_beta_est_all.eps", dpi=300)
                plt.show()
            elif est_beta and est_gamma and not est_alpha and not est_alpha2 and not est_sigma:
                # plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=gamma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/beta_gamma_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_beta and est_sigma and not est_alpha and not est_alpha2 and not est_gamma:
                #plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\theta_{t,3}^N$ (Estimator 1)")
                #plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\theta_{t,3}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 0], 1), label=r"$\theta_{t,3}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 1], 1), label=r"$\theta_{t,3}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                #plt.plot(t, np.mean(all_sigma_t[:, :, 0], 1), label=r"$\eta_{t,1}^N$")
                plt.plot(t, np.mean(all_sigma_ips_t[:, :, 0], 1), label=r"$\eta_{t,1}^N$")
                plt.axhline(y=sigma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/beta_sigma_est_all_ips.eps", dpi=300)
                plt.show()
            elif est_alpha and est_beta and est_gamma and not est_alpha2 and not est_sigma:
                # plt.plot(t, np.mean(all_alpha_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_alpha_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 0], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_alpha_ips_t[:, :, 1], 1), label=r"$\alpha_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=alpha_true, linestyle="--", color="black")
                # plt.plot(t, np.mean(all_beta_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_beta_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 0], 1), label=r"$\beta_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_beta_ips_t[:, :, 1], 1), label=r"$\beta_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=beta_true, linestyle="--", color="black")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (Estimator 1)")
                # plt.plot(t, np.mean(all_gamma_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (Estimator 2)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 0], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 1)")
                plt.plot(t, np.mean(all_gamma_ips_t[:, :, 1], 1), label=r"$\gamma_{t}^N$ (IPS Estimator 2)")
                plt.axhline(y=gamma_true, linestyle="--", color="black")
                plt.legend()
                if save_plots:
                    plt.savefig(path + "/alpha_beta_gamma_est_all_ips.eps", dpi=300)
                plt.show()