"""
Script containing rudimentary function tests.

Author   : Mike Stanley
Created  : Sept 19 2022
Last Mod : Sept 19 2022
"""
import numpy as np
from run_grid_estimation import i_estimate_quantile_at_point
from llr_solvers import exp1_llr


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
