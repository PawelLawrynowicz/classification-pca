import sys
from time import time

import pulp
import numpy as np
import numpy.linalg as la
from pulp import LpProblem, LpStatus, lpSum, LpVariable, LpAffineExpression, PULP_CBC_CMD
from requests import Timeout
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def solve_l1_regression(X, j):
    """
    Solves L1 linear regression problem
    """
    debug = False

    n, m = len(X), len(X[0])
    model = LpProblem(name=f"l1_regression", sense=pulp.LpMinimize)

    b = [LpVariable(name=f"b{i}") for i in range(m)]
    e_p = [LpVariable(name=f"e_p{i}", lowBound=0) for i in range(n)]
    e_m = [LpVariable(name=f"e_m{i}", lowBound=0) for i in range(n)]

    for i in range(n):
        model += (LpAffineExpression([(b[k], X[i][k]) for k in range(m)]) + e_p[i] - e_m[i] == 0, f"minimize for i={i}")
    model += (b[j] == -1, f"dependent variable j = {j}")
    model += lpSum(e_p) + lpSum(e_m)

    model.solve(PULP_CBC_CMD(msg=False, timeLimit=10))
    if debug:
        print(f"status: {model.status}, {LpStatus[model.status]}")
        for var in model.variables():
            print(f"{var.name}: {var.value()}")
        for name, constraint in model.constraints.items():
            print(f"{name}: {constraint.value()}")
    return model.objective.value(), [v.value() for v in b]


def get_V_k(A, V, k):
    """
    Gets k-1 columns from V, corresponding to best values in diagonal matrix A
    """
    num = k - 1
    columns = sorted(np.argpartition(A, -num)[-num:])
    V_k = V[:, columns]
    return V_k


def find_best_fit_subspace(X, m):
    """
    Solves linear programming problem m times for each variable j to find best fit subspace
    and returns its normal vector
    """
    j_star = -1
    R_star = sys.maxsize
    b_star = None
    for j in range(m):
        R, b = solve_l1_regression(X, j)
        if R < R_star:
            R_star = R
            b_star = b
            j_star = j
    return b_star, j_star


def get_modified_id(m, j, b_m):
    """
    Returns modified identity matrix
    """
    I = np.identity(m)
    for l in range(m):
        if j != l:
            I[j, l] = b_m[l] / np.linalg.norm(b_m)
        else:
            I[j, l] = 0
    return I


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X -= mean
    X /= std
    return X


def l1pca(X, num_components):
    """
    refer to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3746759/#R5
    """
    X = standardize(X)
    n, m = len(X), len(X[0])

    X_k = X  # projection of X into k dimensional space
    V = {m + 1: np.identity(m)}  # series of projection matrices
    a = []  # primary components
    for k in range(m, 1, -1):
        b_k, j_k = find_best_fit_subspace(X_k, k)
        I_k = get_modified_id(k, j_k, b_k)
        Z = np.dot(X_k, I_k.T)
        _, A, VT = np.linalg.svd(Z)
        V[k] = get_V_k(A, VT.T, k)

        a_k = V[m + 1]
        for l in range(m, k, -1):
            a_k = np.dot(a_k, V[l])
        # calculate kth primary component
        a_k = np.dot(a_k, b_k) / np.linalg.norm(b_k)
        a.append(a_k)
        X_k = np.dot(Z, V[k])  # project into k-1 space
    a_k = V[m + 1]
    for l in range(m, 1, -1):
        a_k = np.dot(a_k, V[l])
    a.append(a_k.T.flatten())  # calculate 1st primary component
    reduced = np.dot(X, np.array(list(reversed(a)))[:num_components].T)
    return reduced


def run_l1pca_star():
    iris = load_iris()
    iris_pca_reduced = l1pca(iris["data"], 2)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(iris_pca_reduced[:, 0],
               iris_pca_reduced[:, 1], c=iris["target"])
    fig.suptitle('L1-PCA*', fontsize=16)
    plt.show()


def run_pca():
    iris = load_iris()
    iris_pca = PCA(n_components=2)
    iris_pca_reduced = iris_pca.fit_transform(iris["data"])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(iris_pca_reduced[:, 0],
               iris_pca_reduced[:, 1], c=iris["target"])
    fig.suptitle('PCA', fontsize=16)
    plt.show()


if __name__ == '__main__':
    run_pca()
    run_l1pca_star()
