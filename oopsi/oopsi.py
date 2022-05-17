# -*- coding: utf-8 -*-
"""
The Python implementation of the fast-oopsi algorithm

port from the matlab code by jovo:
https://github.com/jovo/oopsi

@author: liubenyuan <liubenyuan AT gmail DOT com>
@date: 2014-04-16
"""
import numpy as np
import numpy.linalg as lp
from scipy.signal import lfilter, detrend
from scipy.sparse import spdiags, eye
from scipy.sparse.linalg.dsolve import linsolve


# generate Fluorescence (F), Calcium2+ (C) and Spikes (N)
def fcn_generate(T, dt=0.02, lam=0.1, tau=1.5, sigma=0.1):
    """
    <input:>
    T     - # of time steps
    dt    - time step size
    lam   - firing rate = lam*dt
    tau   - decay time constant
    sigma - standard derivation of observation noise
    <output:>
    F, C, N
    """
    N = np.random.poisson(lam=lam * dt, size=T)
    gam = 1.0 - dt / tau
    C = lfilter([1.0], [1.0, -gam], N)
    F = C + sigma * np.random.randn(T)  # a=1.0, b=0.0
    return F, C, N


# return mean absolute deviation MAD of F
def oopsi_mad(F):
    """
    python implementation of fast-oopsi, functional blocks are
        fast():
            oopsi_init_par()
            oopsi_est_map()
            while:
                oopsi_est_par()
                oopsi_est_map()
    util functions are,
        oopsi_mad() : mad estimator
        oopsi_m() : generate matric M, where (MC)=n
    this implementation (and jovo's MATLAB) utilize sparse matrix for
    speedup. using scipy.sparse.spdiags and scipy.sparse.eye, Hessian
    equation Hd=g is solved via linsolve.spsolve.
    """
    return np.median(np.abs(F - np.median(F)))


# return sparse bidiagonal matrix
def oopsi_m(gamma, T):
    return spdiags([-gamma * np.ones(T), np.ones(T)], [-1, 0], T, T)


# initialize parameters
def oopsi_init_par(F, dt):
    epsilon = 1e-16
    T = F.shape[0]  # signal length
    F = detrend(F)  # normalize F
    F = (F - F.min()) / (F.max() - F.min()) + epsilon
    a = 1.0  # without scale information
    b = np.median(F)  # baseline, alternate np.percentile(F,0.05)
    lam = 1.0
    gam = 1.0 - dt / 1.0
    sig = oopsi_mad(F) * 1.4826  # median as robust normal scale estimator
    P = {
        "T": T,
        "dt": dt,
        "gamma": gam,
        "alpha": a,
        "beta": b,
        "sigma": sig,
        "lambda": lam,
    }
    return F, P


# map estimator of F
def oopsi_est_map(F, P):
    # extract parameters from dict (turples)
    T, dt, gam, a, b, sig, lam = (
        P[k] for k in ("T", "dt", "gamma", "alpha", "beta", "sigma", "lambda")
    )
    # initialize n,C,llam
    n = 0.01 + np.zeros(T)
    C = lfilter([1.0], [1.0, -gam], n)
    llam = (lam * dt) * np.ones(T)
    # M, H, H1, H2 are 'sparse' matrix, therefore
    # we can directly multiply it with a vector (without np.dot)
    M = oopsi_m(gam, T)
    grad_lnprior = M.T * llam
    H1 = (a**2) / (sig**2) * eye(T)
    z = 1.0  # weight on barrier function
    while z > 1e-13:
        D = F - a * C - b  # residual
        lik = 1 / (2 * (sig**2)) * np.dot(D.T, D)
        post = lik + np.dot(llam.T, n) - z * np.sum(np.log(n))  # calculate L
        s = 1.0
        d = 1.0
        while (lp.norm(d) > 5e-2) and (s > 1e-3):  # conv for z
            glik = -a / (sig**2) * (F - a * C - b)
            g = glik + grad_lnprior - z * (M.T * (1 / n))  # gradient, g
            H2 = spdiags(1 / (n**2), 0, T, T)
            H = H1 + z * (M.T * H2 * M)  # Hessian, H
            d = linsolve.spsolve(H, g)  # direction to step
            # find s
            hit = n / (M * d)  # steps within constraint boundaries
            hit = hit[hit > 0]
            if any(hit < 1):
                s = 0.99 * hit.min()
            else:
                s = 1.0
            # loop over s
            post1 = post + 1.0
            while post1 > post + 1e-7:  # goal: newton step decrease objective
                C1 = C - s * d
                n = M * C1
                D = F - a * C1 - b
                lik1 = 1 / (2 * (sig**2)) * np.dot(D.T, D)
                post1 = lik1 + np.dot(llam.T, n) - z * np.sum(np.log(n))
                s = s / 5.0
                if s < 1e-20:
                    break
            C = C1  # update C
            post = post1  # update post
        z = z / 10.0  # reduce z (sequence of z reductions is arbitrary)
    # clearing n[0],n[1] and normalize n between [0,1]
    n[0:2] = 1e-8
    n = n / n.max()
    return n, C, post


# parameters update for fast-oopsi
def oopsi_est_par(n, C, F, P):
    T, dt, gam, a, b, sig, lam = (
        P[k] for k in ("T", "dt", "gamma", "alpha", "beta", "sigma", "lambda")
    )
    # update
    a = 1.0
    b = np.sum(F - C) / T  # mean bias
    D = F - a * C - b
    mse = np.dot(D.T, D)
    sig = np.sqrt(mse / T)  # RMS of residual error
    lam = T / (dt * np.sum(n))  # inverse of firing rate, n should be normalized
    # packup
    P = {
        "T": T,
        "dt": dt,
        "gamma": gam,
        "alpha": a,
        "beta": b,
        "sigma": sig,
        "lambda": lam,
    }
    return P


# implement fast-oopsi
def fast(F, dt=0.02, iter_max=1, update=True):
    """
    <input:>
    F        - a column vector, fluorescence of a neuron (ROI), Tx1
    dt       - frame sampling interval
    iter_max - maximum iteration
    update   - if we are iterating to estimate parameters
    """
    # initialize parameters
    F, P = oopsi_init_par(F, dt)
    # one-shot Newton-Raphson
    n, C, post = oopsi_est_map(F, P)
    post_max = post
    n_best = n
    C_best = C
    # multiple-shot, if iter_max=0, exit
    ml = np.ones(iter_max)
    ml[0] = post
    # imax = 0
    for i in range(1, iter_max):
        # update parameters based on previous iteration
        if update:
            P = oopsi_est_par(n, C, F, P)
        # update inferred spike train based on new parameters
        n, C, post = oopsi_est_map(F, P)
        if post > post_max:
            n_best = n
            C_best = C
            post_max = post
        # if lik doesnt change much (relatively),or return to a previous state
        ml[i] = post
        if np.abs((ml[i] - ml[i - 1]) / ml[i]) < 1e-3 or any(
            np.abs(ml[:i] - ml[i]) < 1e-5
        ):
            # imax = i
            break
    return n_best, C_best


"""
implement wiener filter,
"""


def wiener(F, dt=0.020, iter_max=20, update=True):
    # normalize
    F = (F - F.mean()) / np.abs(F).max()
    T = F.shape[0]
    gam = 1.0 - dt / 1.0
    M = spdiags([-gam * np.ones(T), np.ones(T)], [-1, 0], T, T)
    C = np.ones(T)
    n = M * C
    lam = 1.0
    llam = lam * dt
    sig = 0.1 * lp.norm(F)  # 0.1 is arbitrary
    #
    D0 = F - C  # we assume a=1.0, b=0.0
    D1 = n - llam
    lik = np.dot(D0.T, D0) / (2 * sig**2) + np.dot(D1.T, D1) / (2 * llam)
    gtol = 1e-4
    #
    for i in range(iter_max):
        # g = -(F-C)/sig**2 + ((M*C).T*M-llam*(M.T*np.ones(T)))/llam
        g = -(F - C) / sig**2 + (M.T * (M * C) - llam * (M.T * np.ones(T))) / llam
        H = eye(T) / sig**2 + M.T * M / llam
        d = linsolve.spsolve(H, g)
        C = C - d
        N = M * C
        #
        old_lik = lik
        D0 = F - C
        D1 = n - llam
        lik = np.dot(D0.T, D0) / (2 * sig**2) + np.dot(D1.T, D1) / (2 * llam)
        if lik <= old_lik - gtol:  # NR step decreases likelihood
            n = N
            if update:
                sig = np.sqrt(np.dot(D0.T, D0) / T)
        else:
            break
    n = n / n.max()
    return n, C


"""
implement discretize, bins can be [threshold] or numOfBins(>=2)
"""


def discretize(F, bins=[0.12], high_pass=True):
    epsilon = 1e-3
    if high_pass:
        v = np.diff(F, axis=0)
    else:
        v = F[1:]
    vmax = v.max() + epsilon
    vmin = v.min() - epsilon
    D = np.zeros(F.shape)
    # pre-allocate the storage (do we need?)
    if np.isscalar(bins):
        binEdges = np.linspace(vmin, vmax, bins + 1)
    else:
        binEdges = np.array(bins)
    D[1:] = np.digitize(v, binEdges)
    D[0] = epsilon
    D = D / D.max()
    return D, v
