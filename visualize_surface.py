"""
Script to visualize these simulated 3d surfaces.

NOTE: assumes the [0,3] with 20 elements grid.

Author   : Mike Stanley
Created  : Sept 8 2022
Last Mod : Sept 29 2022
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_quantile_surface(data, prob, grid_size, grid_ub):
    """
    Quantile Surface. Surface should be BELOW presecribed quantile.
    """
    # reshape the quantile estimates
    quant_est_exp = np.reshape(
        a=data['quantile_ests'], newshape=(grid_size, grid_size), order='C'
    )

    # plot surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    x_grid = np.linspace(0, grid_ub, num=grid_size)
    X, Y = np.meshgrid(x_grid, x_grid)

    # plot the sampled quantile surface
    surf = ax.plot_surface(
        X, Y, quant_est_exp, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # plot the plane for the chi-squared quantile
    ax.plot_surface(
        X, Y, np.ones_like(quant_est_exp) * stats.chi2(df=1).ppf(prob)
    )

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # labels
    ax.set_title(
        r"""
            Sampled %.1f-th Quantiles over grid of true parameters versus
            $\chi^2_1$ %.1f-th Quantile\n
            Dominance means surface is BELOW chi-squared quantile
        """ % (prob * 100, prob * 100)
    )
    plt.tight_layout()
    plt.show()


def plot_probability_surface(data, prob, grid_size, grid_ub):
    """
    Probability surface. Surface should be ABOVE prescribed probability.
    """
    # estimated probabilities on flattened grid
    est_probs_flat_exp1 = np.mean(
        data['llrs'] <= stats.chi2(df=1).ppf(prob), axis=1
    )

    # reshape the probability estimates
    est_probs_exp1 = np.reshape(
        a=est_probs_flat_exp1, newshape=(grid_size, grid_size), order='C'
    )

    # plot surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    x_grid = np.linspace(0, grid_ub, num=grid_size)
    X, Y = np.meshgrid(x_grid, x_grid)

    # plot the plane for the chi-squared quantile
    ax.plot_surface(
        X, Y, np.ones_like(est_probs_exp1) * prob, alpha=0.5
    )

    # plot the sampled quantile surface
    surf = ax.plot_surface(
        X, Y, est_probs_exp1, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # labels
    ax.set_title(
        r"""
            Estimated Prob llr is below %.2f $\chi^2_1$ quantile over grid of true parameters\n
            Dominance means surface is ABOVE chi-squared probability
        """ % prob
    )
    plt.tight_layout()
    plt.show()


def plot_expected_value_surface(data, prob, grid_size, grid_ub):
    """
    Expected value of LLR surface.
    """
    # estimated probabilities on flattened grid
    exp_val_llr_flat = np.mean(
        data['llrs'], axis=1
    )

    # reshape the probability estimates
    exp_val_llr = np.reshape(
        a=exp_val_llr_flat, newshape=(grid_size, grid_size), order='C'
    )

    # plot surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

    x_grid = np.linspace(0, grid_ub, num=grid_size)
    X, Y = np.meshgrid(x_grid, x_grid)

    # plot the plane for the chi-squared quantile
    ax.plot_surface(
        X, Y, np.ones_like(exp_val_llr) * stats.chi2(df=1).mean(), alpha=0.5
    )

    # plot the sampled quantile surface
    surf = ax.plot_surface(
        X, Y, exp_val_llr, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # labels
    ax.set_title(
        r"""
            Expected value of Log-likelihood ratio.
        """
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # read in data
    DATA_PATH = './data/exp7.npz'
    data = np.load(DATA_PATH)

    # plot the experiment settings
    print(data['exp_params_txt'])

    # plotting parameters
    PROB = 0.67
    GRID_SIZE = 20
    GRID_UB = 3

    # plot data
    plot_quantile_surface(
        data=data, prob=PROB, grid_size=GRID_SIZE, grid_ub=GRID_UB
    )
    plot_probability_surface(
        data=data, prob=PROB, grid_size=GRID_SIZE, grid_ub=GRID_UB
    )
    # plot_expected_value_surface(
    #     data=data, prob=PROB, grid_size=GRID_SIZE, grid_ub=GRID_UB
    # )
