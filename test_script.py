"""
Script containing rudimentary function tests.

Author   : Mike Stanley
Created  : Sept 19 2022
Last Mod : Sept 19 2022
"""
import numpy as np
from run_grid_estimation import i_estimate_quantile_at_point
from llr_solvers import exp1_llr, exp3_llr, num_llr
from scipy import stats


def test_i_estimate_quantile_at_point():
    """
    Tests the output type for experiment 1 (h = (0.5 0.5))
    """
    i = 0
    X_TRUE = np.array([1, 1])
    llr = exp1_llr()
    NUM_SAMP = 10
    Q = 0.67

    i, quantile_est, llrs = i_estimate_quantile_at_point(
        i=i,
        x_true=X_TRUE,
        llr=llr,
        num_samp=NUM_SAMP,
        q=Q,
        c_max=20, tol=1e-4
    )

    assert isinstance(i, int)
    assert isinstance(quantile_est, float)
    assert isinstance(llrs, np.ndarray)


def test_exp3_llr(tol=1e-6):
    """
    Test to ensure analytical solution matches numerical solution for Tenorio
    counter example

    tol gauges how different the two solutions are.
    """
    h = np.array([1, -1])
    x_true = np.array([1, 1])
    cvx_llr = num_llr(h=h)
    analytical_llr = exp3_llr()

    # generate some data
    NUM_DATA = 100
    data = x_true + \
        stats.multivariate_normal(np.zeros(2), np.identity(2)).rvs(NUM_DATA)

    cvx_sols = np.zeros(NUM_DATA)
    analytic_sols = np.zeros(NUM_DATA)

    for i in range(NUM_DATA):
        cvx_sols[i] = cvx_llr.compute(y=data[i], x_true=x_true)
        analytic_sols[i] = analytical_llr.compute(y=data[i], x_true=x_true)

    # find the max difference to gauge similarity
    assert np.abs(np.array(cvx_sols) - np.array(analytic_sols)).max() < tol
