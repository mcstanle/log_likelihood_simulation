"""
Functions to estimate quantiles of the log-likelihood ratio distribution.

q-quantile of distribution X ~ F is defined as the smallest t such that
F(t) = P(X leq t) = q.

NOTE: relies on cdf_estimators.py

Author   : Mike Stanley
Created  : Sept 6 2022
Last Mod : Sept 6 2022
"""
from cdf_estimators import cdf_est_linear_interp, interp_function_wrap
import cvxpy as cp
from functools import partial
import numpy as np


def quant_est_coarse(sampled_values, q):
    """
    Estimate q-quantile using ECDF. Called "coarse" because it uses the
    step-wise ECDF.

    Parameters:
        sampled_values (np arr) : iid samples from distribution
        q              (float)  : quantile level

    Returns:
        First sorted datum such that at least q mass is below it
    """
    sorted_vals = np.sort(sampled_values)
    quantile_idx = np.where(
        np.linspace(0, 1, num=sorted_vals.shape[0]) >= q
    )[0][0]
    return sorted_vals[quantile_idx]


def quantile_binary_search(q, cdf_func, c_max=20, tol=1e-4):
    """
    Binary search to find quantile given CDF function.

    Parameters:
        q        (float) : quantile level -- 0 to 1
        cdf_func (func)  : cdf function -- takes one argument
        c_max    (float) : maximum considered quantile
        tol      (float) : search stopping criterion

    Returns:
        Quantile within desired tolerance
    """
    c_i = c_max / 2
    c_i_1 = c_max / 2
    c_min_i = 0
    c_max_i = c_max
    change = 1000
    while change > tol:
        prob_i = cdf_func(c_i)
        c_i_1 = c_i

        if prob_i > q:
            c_max_i = c_i
            c_i = (c_i + c_min_i) / 2
        else:
            c_min_i = c_i
            c_i = (c_max_i + c_i) / 2

        change = np.abs(c_i - c_i_1)

    return c_i


def opt_llr(mu, y, h, verbose=False):
    """
    Optimize the log-likelihood ratio

    Parameters:
        mu      (float)  : level set of functional
        y       (np arr) : sampled data
        h       (np arr) : functional of interest
        verbose (bool)   : toggle cvxpy optimizer output

    Return:
        log-likelihood ratio for given data
    """
    n = h.shape[0]
    x_null = cp.Variable(n)
    x_alt = cp.Variable(n)

    # null/alt hypothesis optimization
    prob_null = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - x_null)),
        constraints=[
            h @ x_null == mu,
            x_null >= 0
        ]
    )
    prob_alt = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - x_alt)),
        constraints=[
            x_alt >= 0
        ]
    )

    # solve the optimizations
    opt_null = prob_null.solve(verbose=verbose)
    opt_alt = prob_alt.solve(verbose=verbose)

    return opt_null - opt_alt


def estimate_quantile_at_point(
    x_true, h, noise_distr, num_samp, q, c_max, tol
):
    """
    Takes a true parameter value and estimates q-quantile.

    NOTE: the noise distribution needs to be defined outside.

    Parameters:
        x_true      (np arr)      : true parameter value
        h           (np arr)      : functional of interest
        noise_distr (scipy distr) : multivariate error distribution
        num_samp    (int)         : number of data samples
        q           (float)       : quantile (0, 1)
        c_max       (float)       : maximum considered quantile
        tol         (float)       : search stopping criterion

    Returns:
        quantile_est (float) : estimate of quantile
        sampled_data (np arr) : truth + noise
        llrs         (np arr) : log-likelihood ratios for each data draw
    """
    # generate data
    sampled_data = x_true + noise_distr.rvs(num_samp)

    # compute true functional value
    mu_true = np.dot(h, x_true)

    # compute llrs
    llrs = np.zeros(num_samp)
    for i in range(num_samp):
        llrs[i] = opt_llr(mu=mu_true, y=sampled_data[i], h=h)

    # define an approx CDF from the above sampled data
    cdf_approx, interp_xs = cdf_est_linear_interp(llrs)
    cdf_approx = partial(
        interp_function_wrap,
        interp_func=cdf_approx, interp_xs=interp_xs
    )

    # estimate quantile
    quantile_est = quantile_binary_search(
        q=q,
        cdf_func=cdf_approx,
        c_max=20,
        tol=1e-4
    )

    return quantile_est, sampled_data, llrs
