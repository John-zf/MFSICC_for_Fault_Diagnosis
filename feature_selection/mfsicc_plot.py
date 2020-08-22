# -*- coding: utf-8 -*-
"""
Created on 2019/11/20 9:20

@author: John_Fengz
"""

import numpy as np
from scipy.linalg import solve_sylvester
from sklearn.preprocessing import OneHotEncoder
import copy


def _cal_l21norm(matrix):
    """
    :param matrix: array-like (numpy.array)
    :return: value of l2,1 norm
    """
    return np.sum(np.sqrt(np.sum(matrix * matrix, 1)))


class MFSICC:
    def __init__(self, lamb1=0.1, alpha=0.5, lamb2=0.1,
                 n_iter=50, random_state=42):
        """
        Parameters
        ----------
        lamb1: float, optional (default=0.1)
            Penalty factor to control view complementarity.

        alpha: float, optional (default=0.5)
            Ratio to control local and global sparsity.

        lamb2: float, optional (default=0.1)
            Penalty factor to control view consensus.

        n_iter: int, optional (default=50)

        random_state: int, RandomState instance or None, optional (default=42)
        """
        self.lamb1 = lamb1
        self.alpha = alpha
        self.lamb2 = lamb2
        self.n_iter = n_iter
        self.random_state = random_state

        self.weight_matrix = None
        self.obj = np.zeros(self.n_iter)

    def fit(self, x, y, group):
        # np.random.seed(self.random_state)
        eps = 0.00001
        x = x.T
        y = np.array(OneHotEncoder(categories='auto', sparse=False)
                     .fit_transform(y[:, np.newaxis]))

        dim, n = x.shape
        class_num = y.shape[1]
        group_set = set(group)

        w = np.random.randn(dim, class_num)
        w = np.ones((dim, class_num)) * self.random_state
        for i in range(self.n_iter):
            sum1 = 0
            for count, k in enumerate(group_set):
                idx = np.argwhere(group == k).T[0]
                w_k = w[idx, :]
                x_k = x[idx, :]

                a = w_k * w_k
                p_k = np.diag(0.5 / np.sqrt(np.sum(a, 0) + eps))
                q_k = np.diag(0.5 / np.sqrt(np.sum(a, 1) + eps))

                # calculate the second term of C
                sum0 = np.zeros(w_k.shape)
                for j in group_set:
                    if k == j:
                        continue
                    else:
                        idx1 = np.argwhere(group == j).T[0]
                        w_k_ = w[idx1, :]
                        x_k_ = x[idx1, :]
                        sum0 = sum0 + x_k @ x_k_.T @ w_k_

                a = (1 + self.lamb2) * (x_k @ x_k.T) + self.alpha * self.lamb1 * q_k
                b = (1 - self.alpha) * self.lamb2 * p_k
                c = x_k @ y + self.lamb2 * sum0
                # Solve Sylvester matrix equation
                w_k_s = solve_sylvester(a, b, c)
                w[idx, :] = w_k_s
                sum1 += _cal_l21norm(w_k_s.T)

            # Calculate group consitence
            sum2 = 0
            group_set_copy = copy.deepcopy(group_set)
            for count1, p in enumerate(group_set):
                idx_p = np.argwhere(group == p).T[0]
                w_p = w[idx_p, :]
                x_p = x[idx_p, :]
                group_set_copy.remove(p)
                for q in group_set_copy:
                    idx_q = np.argwhere(group == q).T[0]
                    w_q = w[idx_q, :]
                    x_q = x[idx_q, :]
                    temp = x_p.T @ w_p - x_q.T @ w_q
                    sum2 += _cal_l21norm(temp)

            self.obj[i] = _cal_l21norm(x.T @ w) \
                          + (1 - self.alpha) * self.lamb1 * sum1 \
                          + self.alpha * self.lamb1 * _cal_l21norm(w) \
                          + self.lamb2 * sum2
        self.weight_matrix = w

    def get_objectives(self):
        return self.obj

    def get_weight_matrix(self):
        return self.weight_matrix

    def get_feature_weights(self):
        return np.sum(self.weight_matrix * self.weight_matrix, 1)

    def feature_ranking(self):
        return np.argsort(-np.sum(self.weight_matrix * self.weight_matrix, 1))
