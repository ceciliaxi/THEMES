"""Bayesian Reward Distribution. """

# Authors: Hamoon Azizsoltani <hazizso@ncsu.edu>
#
# License: BSD 3 clause

import warnings
from operator import itemgetter
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from scipy.optimize import fmin_l_bfgs_b
import time


class BayesianRewardDistribution(BaseEstimator, RegressorMixin):
    """Bayesian Reward Distribution (BRD).
    The bayesian reward distribution algorithm can ba used to distribute the delayed reward inside the trajectories


    """
    def __init__(self, kernel=None, alpha=1e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, copy_X_train=True,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, R, D):
        """Fit Bayesian Reward Distribution model.

        Parameters
        ----------


        Returns
        -------
        self : returns an instance of self.
        """

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        elif not isinstance(self.kernel, np.ndarray):
            self.kernel_ = clone(self.kernel)

        # self._rng = check_random_state(self.random_state)
        # X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != R.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], R.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.R_train_ = np.copy(R) if self.copy_X_train else R

        # Precompute quantities required for predictions which are independent
        # of actual query points
        if not isinstance(self.kernel, np.ndarray):
            K_D_T = np.dot(self.kernel_(self.X_train_), np.transpose(D))
        else:
            K_D_T = np.dot(self.kernel, np.transpose(D))

        D_K_D_T = np.float64(np.dot(D, K_D_T))
        D_K_D_T[np.diag_indices_from(D_K_D_T)] += self.alpha
        try:
            self.L_ = cholesky(D_K_D_T, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "Bayesian estimator."
                        % self.kernel_,) + exc.args
            raise

        self.alpha_ = cho_solve((self.L_, True), self.R_train_)  # Line 3
        return self

    def predict(self, X, D, K=None, return_std=False, return_cov=False):
        """Predict using the Bayesian Reward Distribution model
        Parameters
        ----------


        Returns
        -------

        """
        if not isinstance(K, np.ndarray):
            K = self.kernel_(X, self.X_train_)

        K_D_T = np.dot(K, np.transpose(D))
        y_mean = K_D_T.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
            return y_mean, y_cov
        elif return_std:
            # cache result of K_inv computation
            if self._K_inv is None:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T,
                                         np.eye(self.L_.shape[0]))
                self._K_inv = L_inv.dot(L_inv.T)

            # Compute variance of predictive distribution
            y_var = self.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(K_trans, self._K_inv), K_trans)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean
