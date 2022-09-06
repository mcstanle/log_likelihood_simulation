"""
Functions to estimate CDFs.

Author   : Mike Stanley
Created  : Sept 6 2022
Last Mod : Sept 6 2022
"""
import numpy as np
from scipy.interpolate import interp1d


def interp_function_wrap(t, interp_func, interp_xs):
    """
    Wrapper around a CDF function that doesn't account for the left and right
    extremes.

    Parameters:
        t           (float)    : value at which to evaluate cdf
        interp_func (function) : interpolated defined from min to max data
        interp_xs   (np arr)   : values on which the interpolation was made

    Returns:
        cdf evaluated at t
    """
    if t <= interp_xs[0]:
        return 0
    elif t >= interp_xs[-1]:
        return 1
    else:
        return interp_func(t)


def cdf_est_linear_interp(sampled_data):
    """
    Linear interpolation of ECDF. The step midpoints are the interpolated
    values.

    Parameters:
        sampled_data (np arr) : iid draws from distribution

    Returns:
        interp_func (func)   : Callable Linearly interpolated ECDF
        interp_xs   (np arr) : x values used for interpolation
    """
    # order the sampled values
    ordered_data = np.sort(sampled_data)

    # make sure there is only one zero value
    zero_idxs = np.where(ordered_data == 0)[0]
    if len(zero_idxs) > 0:
        ordered_data_1zero = ordered_data[zero_idxs[-1]:]
    else:
        ordered_data_1zero = ordered_data.copy()

    # find the midpoints
    md_pts = 0.5 * (ordered_data_1zero[:-1] + ordered_data_1zero[1:])

    # made a new set of values to interpolate
    # first data point is the true smallest
    # last data point is the true largest
    interp_xs = np.zeros(len(ordered_data_1zero) + 1)
    interp_xs[0] = ordered_data_1zero[0]
    interp_xs[1:-1] = md_pts
    interp_xs[-1] = ordered_data_1zero[-1]

    # define the interpolation function
    interp_func = interp1d(
        x=interp_xs,
        y=np.linspace(0, 1, num=len(interp_xs)), kind='linear'
    )

    return interp_func, interp_xs
