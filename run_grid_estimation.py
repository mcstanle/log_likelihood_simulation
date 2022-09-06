"""
Estimates quantiles on a defined grid of points.

Parallelizable.

Author   : Mike Stanley
Created  : Sept 6 2022
Last Mod : Sept 6 2022
"""
import multiprocessing as mp
import numpy as np
from quantile_estimators import estimate_quantile_at_point
from scipy import stats
from time import time
import os


def i_estimate_quantile_at_point(
    i, x_true, h, noise_distr, num_samp, q, c_max, tol
):
    """
    Wrapper around estimate_quantile_at_point so that the index can be
    included when results need to be recompiled.

    Parameters:
        i           (int)         : index number of current grid point
        x_true      (np arr)      : true parameter value
        h           (np arr)      : functional of interest
        noise_distr (scipy distr) : multivariate error distribution
        num_samp    (int)         : number of data samples
        q           (float)       : quantile (0, 1)
        c_max       (float)       : maximum considered quantile
        tol         (float)       : search stopping criterion

    Returns:
        i            (int)    : index number of current grid point
        quantile_est (float)  : estimate of quantile
        sampled_data (np arr) : truth + noise
        llrs         (np arr) : log-likelihood ratios for each data draw
    """
    quantile_est, sampled_data, llrs = estimate_quantile_at_point(
        x_true=x_true,
        h=h,
        noise_distr=noise_distr,
        num_samp=num_samp,
        q=q,
        c_max=c_max,
        tol=tol
    )

    return i, quantile_est, sampled_data, llrs


def parallel_quantile_est(
    grid_flat, h, noise_distr, num_samp, q, c_max, tol, num_cpu=None
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
        h           (np arr)      : functional of interest (d)
        noise_distr (scipy distr) : multivariate error distribution
        num_samp    (int)         : number of data samples (n)
        q           (float)       : quantile (0, 1)
        c_max       (float)       : maximum considered quantile
        tol         (float)       : search stopping criterion
        num_cpu     (int)         : number of CPUs over which to parallelize

    Returns:
        quantile_ests (np arr) : estimated quantiles for each grid point
        sampled_datas (np arr) : all sampled data for each grid point est
        llrs          (np arr) : log-lr for each data draw
    """
    if num_cpu:
        pool = mp.Pool(num_cpu)
    else:
        pool = mp.Pool(mp.cpu_count())

    print('Number of available CPUs: %i' % mp.cpu_count())

    # storage for run_pipeline() output
    output_data = []

    def collect_data(data):
        output_data.append(data)

    for i in range(grid_flat.shape[0]):
        pool.apply_async(
            i_estimate_quantile_at_point,
            args=(
                i, grid_flat[i],
                h, noise_distr, num_samp, q, c_max, tol
            ),
            callback=collect_data
        )

    pool.close()
    pool.join()

    output_data.sort(key=lambda x: x[0])
    quantile_ests = np.array([_[1] for _ in output_data])  # (g**2,)
    sampled_datas = np.array([_[2] for _ in output_data])  # (g**2, n, d)
    llrs = np.array([_[3] for _ in output_data])           # (g**2, n)

    return quantile_ests, sampled_datas, llrs


if __name__ == "__main__":

    # define the grid
    NUM_GRID = 10
    GRID_LB = 0
    GRID_UB = 3
    x_1_grid = np.linspace(GRID_LB, GRID_UB, num=NUM_GRID)
    x_2_grid = np.linspace(GRID_LB, GRID_UB, num=NUM_GRID)

    count = 0
    grid_flat = np.zeros(shape=(NUM_GRID ** 2, 2))
    for i in range(NUM_GRID):
        for j in range(NUM_GRID):
            grid_flat[count, :] = [x_1_grid[i], x_2_grid[j]]

        count += 1

    # define the parameters for the simulations
    h = np.array([0.5, 0.5])
    noise_distr = stats.multivariate_normal(
        mean=np.zeros(2),
        cov=np.identity(2)
    )
    NUM_SAMP = 1000
    Q = 0.67
    C_MAX = 20
    TOL = 1e-4
    NUM_CPU = 8
    OUTPUT_FILE_NM = 'exp0.npz'

    # constract text file with experiment parameters
    exp_params_txt = "NUM_GRID = %i\n" % NUM_GRID
    exp_params_txt += "GRID_LB = %i | GRID_UB = %i\n" % (GRID_LB, GRID_UB)
    exp_params_txt += "h = %s\n" % str(h)
    exp_params_txt += "NUM_SAMP = %i\n" % NUM_SAMP
    exp_params_txt += "q = %.2f | c_max = %.2f | tol = %s\n" % (
        Q, C_MAX, str(TOL)
    )
    if NUM_CPU:
        exp_params_txt += "NUM_CPU = %i" % NUM_CPU
    else:
        exp_params_txt += "NUM_CPU = %i" % mp.cpu_count()

    # estimate quantiles
    START = time()
    quantile_ests, sampled_datas, llrs = parallel_quantile_est(
        grid_flat=grid_flat,
        h=h,
        noise_distr=noise_distr,
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
    if os.path.exits('./data'):
        SAVE_PATH = './data/%s' % OUTPUT_FILE_NM
    else:
        SAVE_PATH = './%s' % OUTPUT_FILE_NM

    np.savez(
        file='./data/%s' % OUTPUT_FILE_NM,
        exp_params_txt=exp_params_txt,
        quantile_ests=quantile_ests,
        sampled_datas=sampled_datas,
        llrs=llrs
    )