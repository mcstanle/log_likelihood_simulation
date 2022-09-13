"""
Log-likelihood ratio solvers, both analytical and numerical.

For some functionals, the solution is analytically tractible. It is useful to
compare these solutions to the numerical ones achieved with cvxpy. See below
for a key, linking experimental setups with functionals.

NOTE: all of the analytical solutions assume the non-negative orthant as the
convex cone.

1. exp1 -- h = (0.5 0.5)

Author   : Mike Stanley
Created  : 11 Sept 2022
Last Mod : 13 Sept 2022
"""
import cvxpy as cp
import numpy as np


class opt_llr:
    """ Class of log-likelihood ratio solvers """
    def __init__(self, mu, h):
        self.mu = mu
        self.h = h

    def compute(self, **kwargs):
        """
        compute -2 log Lambda(mu, y)
        """
        raise NotImplementedError("Implement in subclass")


class exp1_llr(opt_llr):
    """ Child class for h = (0.5 0.5) -- explicit solution exists """
    def __init__(self, mu, h, x_true):
        super().__init__(mu, h)
        self.x_true = x_true
        self.h = np.array([0.5, 0.5])  # overwrite to particular functional
        self.mu = np.dot(self.h, self.x_true)  # overwrite with true val

    def compute(self, y):
        """
        h = (0.5 0.5)

        To make the solving easier, I use the S_i and Q_j terminology.
        """
        # find solutions for regions
        quad_sols = {
            'Q1': 0,
            'Q2': y[0] ** 2,
            'Q3': np.dot(y, y),
            'Q4': y[1] ** 2
        }
        S_sols = {
            'S1': y[0] ** 2 + (y[1] - 2 * self.mu) ** 2,
            'S2': np.linalg.norm(2 * (self.mu - np.dot(self.h, y)) * self.h) ** 2,
            'S3': (y[0] - 2 * self.mu) ** 2 + y[1] ** 2
        }

        # determine quadrant of y
        y_1_nn = y[0] >= 0
        y_2_nn = y[1] >= 0
        if y_1_nn and y_2_nn:
            quad = 'Q1'
        elif not y_1_nn and y_2_nn:
            quad = 'Q2'
        elif not y_1_nn and not y_2_nn:
            quad = 'Q3'
        else:
            quad = 'Q4'

        # determine S region
        if y[1] > y[0] + 2 * self.mu:
            S_reg = 'S1'
        elif y[1] < y[0] - 2 * self.mu:
            S_reg = 'S3'
        else:
            S_reg = 'S2'

        return S_sols[S_reg] - quad_sols[quad]


class num_llr(opt_llr):
    """ cvxpy numerical solver """
    def __init__(self, mu, h):
        super().__init__(mu, h)

    def compute(self, y, verbose=False):
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
        n = self.h.shape[0]
        x_null = cp.Variable(n)
        x_alt = cp.Variable(n)

        # null/alt hypothesis optimization
        prob_null = cp.Problem(
            objective=cp.Minimize(cp.sum_squares(y - x_null)),
            constraints=[
                self.h @ x_null == self.mu,
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
