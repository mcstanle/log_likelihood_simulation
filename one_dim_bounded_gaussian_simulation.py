"""
Run the one-dimensional bounded Gaussian example.

NOTE: based on ~/Research/strict_boundds/theoretical_exploration/
    one_dimensional_non_negative_gaussian.ipynb

NOTE: parallelization code is based on ./run_grid_estimation.py

Author   : Mike Stanley
Created  : Nov 17 2022
Last Mod : Nov 17 2022
"""
import cvxpy as cp
import multiprocessing as mp
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def cdf_analy(mu, c):
    """ The analytical cdf for Gaussian with non-negative mean """
    if c < mu ** 2:
        return stats.chi2(df=1).cdf(c)
    else:
        a = 0 if mu == 0. else stats.norm.cdf((-mu**2 - c)/(2 * mu))
        return stats.norm.cdf(np.sqrt(c)) - a


def quantile_binary_search(mu, c_max=20, alpha=0.05, tol=1e-8):
    """ find 1 - alpha quantile using cdf_analy() """
    c_i = c_max / 2
    c_i_1 = c_max / 2
    c_min_i = 0
    c_max_i = c_max
    change = 1000
    while change > tol:
        prob_i = cdf_analy(mu=mu, c=c_i)
        c_i_1 = c_i

        if prob_i > 1 - alpha:
            c_max_i = c_i
            c_i = (c_i + c_min_i) / 2
        else:
            c_min_i = c_i
            c_i = (c_max_i + c_i) / 2

        change = np.abs(c_i - c_i_1)

    return c_i


def osb_int_q(y):
    """
    c is the quantile

    One-dimensional ONLY

    Using scipy because need sub computation done with q_alpha
    """
    x_lb = cp.Variable(1)
    x_ub = cp.Variable(1)

    # this is a hack to get this to work
    x_lb.value = 0.25
    x_ub.value = 0.25

    # solve the slack optimization
    x_slack = cp.Variable(1)
    prob_slack = cp.Problem(
        objective=cp.Minimize(cp.square(y - x_slack)),
        constraints=[x_slack >= 0]
    )
    opt_slack = prob_slack.solve()

    # set up the optimization problems
    prob_lb = cp.Problem(
        objective=cp.Minimize(x_lb),
        constraints=[
            cp.square(y - x_lb) - opt_slack <= quantile_binary_search(
                mu=x_lb.value
            ),
            x_lb >= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(-x_ub),
        constraints=[
            cp.square(y - x_ub) - opt_slack <= quantile_binary_search(
                mu=x_ub.value
            ),
            x_ub >= 0
        ]
    )

    # solve the optimizations
    opt_lb = prob_lb.solve()
    opt_ub = prob_ub.solve()

    return opt_lb, -opt_ub


def osb_int_q_scipy(i, y, x0=0.25):
    """
    One-dimensional ONLY

    Using scipy because need sub computation done with q_alpha

    Parameters:
    -----------
        i  (int)   : index of the current interval
        y  (float) : 1d data sample
        x0 (float) : starting position for the optimizer

    Returns:
    --------
        (i, [res_lb.x, res_ib.x])
    """
    # solve the slack optimization
    x_slack = cp.Variable(1)
    prob_slack = cp.Problem(
        objective=cp.Minimize(cp.square(y - x_slack)),
        constraints=[x_slack >= 0]
    )
    opt_slack = prob_slack.solve()

    # define constraints
    cons = (
        {
            'type': 'ineq',
            'fun': lambda x: quantile_binary_search(mu=x) - np.square(y - x) + opt_slack
        }
    )
    bnds = [[0, np.inf]]
    res_lb = minimize(fun=lambda x: x, x0=x0, bounds=bnds, constraints=cons)
    res_ub = minimize(fun=lambda x: -x, x0=x0, bounds=bnds, constraints=cons)

    return i, [res_lb.x, res_ub.x]


def parallel_interval_est(
    data, x0, num_cpu=None
):
    """
    Parallelize osb_int_q_scipy() over num_cpu.

    Parameters:
        data    (np arr) : 1d array of sampled data values
        x0      (float)  : starting position for optimization
        num_cpu (int)    : number of CPUs over which to parallelize

    Returns:
        output_intervals_final (np arr) : computed intervals
    """
    if num_cpu:
        pool = mp.Pool(num_cpu)
    else:
        pool = mp.Pool(mp.cpu_count())

    print('Number of available CPUs: %i' % mp.cpu_count())

    # storage for interval estimation output
    output_intervals = [None] * data.shape[0]

    def collect_interval(output):
        idx = output[0]
        output_intervals[idx] = output

    for i in range(data.shape[0]):
        pool.apply_async(
            osb_int_q_scipy,
            args=(
                i, data[i], x0
            ),
            callback=collect_interval
        )

    pool.close()
    pool.join()

    output_intervals.sort(key=lambda x: x[0])
    output_intervals_final = np.array([_[1] for _ in output_intervals])

    return output_intervals_final


if __name__ == "__main__":

    # set file paths
    BASE_PATH = '/home/mcstanle/log_likelihood_simulation'
    SAVE_PATH = BASE_PATH + '/data/non_negative_gaussian'
    SAVE_PATH += '/x_star_point00.npy'

    # sample data
    x_star = 0.0
    noise_level = 1
    np.random.seed(1)
    N = 5000  # number of samples
    y = stats.norm(loc=x_star, scale=noise_level).rvs(N)

    # compute the intervals
    X0 = 0.25
    output_intervals = parallel_interval_est(
        data=y, x0=X0, num_cpu=None
    )

    # save intervals
    with open(SAVE_PATH, 'wb') as f:
        np.save(file=f, arr=output_intervals)
