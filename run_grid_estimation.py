"""
Estimates quantiles on a defined grid of points.

Parallelizable.

NOTE: bisecting mode is when true parameter values are only taken on the y=x
line. This setup matches the Tenorio (2008) example.

NOTE: to make sure the right solver is being used, look at the SET SOLVER
portion of the main code.

Author   : Mike Stanley
Created  : Sept 6 2022
Last Mod : Aug 29 2023
"""
from llr_solvers import exp3_llr, num_llr  # exp1_llr
import multiprocessing as mp
import numpy as np
from quantile_estimators import estimate_quantile_at_point
from time import time
import os


def i_estimate_quantile_at_point(
    i, x_true, llr, num_samp, q, c_max, tol
):
    """
    Wrapper around estimate_quantile_at_point so that the index can be
    included when results need to be recompiled.

    Parameters:
        i           (int)         : index number of current grid point
        x_true      (np arr)      : true parameter value
        llr         (opt_llr obj) : llr object (see llr_solvers.py)
        num_samp    (int)         : number of data samples
        q           (float)       : quantile (0, 1)
        c_max       (float)       : maximum considered quantile
        tol         (float)       : search stopping criterion

    Returns:
        i            (int)    : index number of current grid point
        quantile_est (float)  : estimate of quantile
        # sampled_data (np arr) : truth + noise
        llrs         (np arr) : log-likelihood ratios for each data draw
    """
    quantile_est, sampled_data, llrs = estimate_quantile_at_point(
        x_true=x_true,
        llr=llr,
        num_samp=num_samp,
        q=q,
        c_max=c_max,
        tol=tol,
        seed=i
    )

    return i, quantile_est, llrs


def parallel_quantile_est(
    grid_flat, llr, num_samp, q, c_max, tol, num_cpu=None
):
    """
    Given the flattened list of parameters over which to simulate, parallelize
    the estimate_quantile_at_point function.

    Dimensions:
        g : number of grid cells in each dimension
        d : parameter dimension (2)
        n : number of samples in each quantile estimation

    Parameters:
        grid_flat   (np arr)      : 2d parameter combos (g**2, d)
        llr         (opt_llr obj) : llr object (see llr_solvers.py)
        num_samp    (int)         : number of data samples (n)
        q           (float)       : quantile (0, 1)
        c_max       (float)       : maximum considered quantile
        tol         (float)       : search stopping criterion
        num_cpu     (int)         : number of CPUs over which to parallelize

    Returns:
        quantile_ests (np arr) : estimated quantiles for each grid point
        # sampled_datas (np arr) : all sampled data for each grid point est
        llrs          (np arr) : log-lr for each data draw
    """
    if num_cpu:
        pool = mp.Pool(num_cpu)
    else:
        pool = mp.Pool(mp.cpu_count())

    print('Number of available CPUs: %i' % mp.cpu_count())

    # storage for run_pipeline() output
    SUM_GRID = grid_flat.shape[0]
    output_data = [None] * SUM_GRID

    def collect_data(data):
        idx = data[0]
        output_data[idx] = data

    for i in range(SUM_GRID):
        pool.apply_async(
            i_estimate_quantile_at_point,
            args=(
                i, grid_flat[i],
                llr, num_samp, q, c_max, tol
            ),
            callback=collect_data
        )

    pool.close()
    pool.join()

    output_data.sort(key=lambda x: x[0])
    quantile_ests = np.array([_[1] for _ in output_data])  # (g**2,)
    # sampled_datas = np.array([_[2] for _ in output_data])  # (g**2, n, d)
    llrs = np.array([_[2] for _ in output_data])           # (g**2, n)

    return quantile_ests, llrs


if __name__ == "__main__":

    # bisecting mode -- see above for description
    BISECTING_MODE = False

    # SET SOLVER
    ANALYTICAL_SOLVER = True
    h = np.array([1, -1])  # manually defined h for numerical solver
    # h = np.array([0.5, 0.5])  # exp1
    # h = np.array([0.25, 0.75])  # exp2
    h = np.array([1, -1])  # exp3
    if ANALYTICAL_SOLVER:

        # select the right analytical solver!
        # llr = exp1_llr()
        llr = exp3_llr()
        assert np.array_equiv(llr.h, h)
        print('Using analytical solver for h = %s' % str(h))
    else:
        llr = num_llr(h=h)
        print('Using cvxpy solver')

    # define the grid
    NUM_GRID_X1 = 40
    NUM_GRID_X2 = 40
    GRID_X1_LB = -3
    GRID_X1_UB = 0
    GRID_X2_LB = -3
    GRID_X2_UB = 0
    BASE = np.exp(1)
    x_1_grid = np.power(
        np.ones(NUM_GRID_X1) * BASE,
        np.linspace(GRID_X1_LB, GRID_X1_UB, num=NUM_GRID_X1)
    )
    x_2_grid = np.power(
        np.ones(NUM_GRID_X2) * BASE,
        np.linspace(GRID_X2_LB, GRID_X2_UB, num=NUM_GRID_X2)
    )

    if BISECTING_MODE:
        grid_flat = np.zeros(shape=(NUM_GRID_X1, 2))
        for i in range(NUM_GRID_X1):
            grid_flat[i, :] = [x_1_grid[i], x_1_grid[i]]
    else:
        count = 0
        grid_flat = np.zeros(shape=(NUM_GRID_X1 * NUM_GRID_X2, 2))
        for i in range(NUM_GRID_X1):
            for j in range(NUM_GRID_X2):
                grid_flat[count, :] = [x_1_grid[i], x_2_grid[j]]

                count += 1

    # define the parameters for the simulations
    NUM_SAMP = 50000
    Q = 0.95
    C_MAX = 20
    TOL = 1e-4
    NUM_CPU = None
    OUTPUT_FILE_NM = 'exp13.npz'

    # constract text file with experiment parameters
    exp_params_txt = "NUM_GRID_X1 = %i\n" % NUM_GRID_X1
    exp_params_txt += "NUM_GRID_X2 = %i\n" % NUM_GRID_X2
    exp_params_txt += "GRID_X1_LB = %i | GRID_X1_UB = %i\n" % (
        GRID_X1_LB, GRID_X1_UB
    )
    exp_params_txt += "GRID_X2_LB = %i | GRID_21_UB = %i\n" % (
        GRID_X2_LB, GRID_X2_UB
    )
    exp_params_txt += "h = %s\n" % str(h)
    exp_params_txt += "NUM_SAMP = %i\n" % NUM_SAMP
    exp_params_txt += "q = %.2f | c_max = %.2f | tol = %s\n" % (
        Q, C_MAX, str(TOL)
    )
    exp_params_txt += "BISECTING_MODE = %s\n" % str(BISECTING_MODE)
    exp_params_txt += "ANALYTICAL SOLVER = %s\n" % str(ANALYTICAL_SOLVER)
    if NUM_CPU:
        exp_params_txt += "NUM_CPU = %i" % NUM_CPU
    else:
        exp_params_txt += "NUM_CPU = %i" % mp.cpu_count()

    # estimate quantiles
    START = time()
    quantile_ests, llrs = parallel_quantile_est(
        grid_flat=grid_flat,
        llr=llr,
        num_samp=NUM_SAMP,
        q=Q,
        c_max=C_MAX,
        tol=TOL,
        num_cpu=NUM_CPU
    )
    END = time()

    print(
        'Time to estimate quantiles on grid: %.4f mins' % ((END - START) / 60)
    )

    # save the above data
    if os.path.exists('./data'):
        SAVE_PATH = './data/%s' % OUTPUT_FILE_NM
    else:
        SAVE_PATH = './%s' % OUTPUT_FILE_NM

    np.savez(
        file=SAVE_PATH,
        # grid_flat=grid_flat,
        exp_params_txt=exp_params_txt,
        quantile_ests=quantile_ests,
        # sampled_datas=sampled_datas,
        # llrs=llrs
    )
