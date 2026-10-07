"""
DYNOTEARS (Pamfil et al., AISTATS 2020): score-based structure learning for a
linear structural vector-autoregressive model

    X_t = X_t W + [X_{t-1} ... X_{t-p}] A + noise

W (d x d) holds the same-slice edges and must be acyclic, A (p*d x d) the
lagged edges. Solved as in the paper: least squares + L1 penalties, acyclicity
of W enforced by an augmented Lagrangian on h(W) = tr(exp(W o W)) - d, each
sub-problem solved with L-BFGS-B on the positive / negative parts of (W, A).

Implemented here because causalnex (the reference implementation) does not
install on Python 3.12. The loss only needs second moments, so the cost per
iteration does not depend on the number of rows.
"""
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt


def dynotears(X, Xlag, lambda_w=0.05, lambda_a=0.05, max_iter=100,
              h_tol=1e-8, rho_max=1e16, w_threshold=0.0,
              forbid_w=None, forbid_a=None):
    """
    X:    (n, d)    values at time t (standardise the columns first)
    Xlag: (n, p*d)  values at t-1, ..., t-p (same column order, lag 1 first)
    forbid_w / forbid_a: boolean masks, True where an edge must stay at zero
        (entry [i, j] is the edge i -> j).

    Returns W (d, d), A (p*d, d); entries below w_threshold are set to zero.
    """
    n, d = X.shape
    pd_ = Xlag.shape[1]

    # second moments
    Sxx = X.T @ X / n
    Sxl = X.T @ Xlag / n          # d x pd
    Sll = Xlag.T @ Xlag / n       # pd x pd

    fw = np.eye(d, dtype=bool) if forbid_w is None else (forbid_w | np.eye(d, dtype=bool))
    fa = np.zeros((pd_, d), dtype=bool) if forbid_a is None else forbid_a

    n_w, n_a = d * d, pd_ * d

    def unpack(v):
        W = (v[:n_w] - v[n_w:2 * n_w]).reshape(d, d)
        A = (v[2 * n_w:2 * n_w + n_a] - v[2 * n_w + n_a:]).reshape(pd_, d)
        return W, A

    def h_func(W):
        E = sla.expm(W * W)
        return np.trace(E) - d, E.T * 2 * W

    def func(v, rho, alpha):
        W, A = unpack(v)
        # 0.5/n * || X - XW - Xlag A ||_F^2, expanded on the moment matrices
        IW = np.eye(d) - W
        loss = 0.5 * (np.trace(IW.T @ Sxx @ IW)
                      - 2 * np.trace(A.T @ Sxl.T @ IW)
                      + np.trace(A.T @ Sll @ A))
        gW = -(Sxx @ IW) + Sxl @ A
        gA = -(Sxl.T @ IW) + Sll @ A
        h, gh = h_func(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda_w * v[:2 * n_w].sum() \
            + lambda_a * v[2 * n_w:].sum()
        gW = gW + (rho * h + alpha) * gh
        g = np.concatenate([gW.ravel() + lambda_w, -gW.ravel() + lambda_w,
                            gA.ravel() + lambda_a, -gA.ravel() + lambda_a])
        return obj, g

    bounds = ([(0, 0) if f else (0, None) for f in fw.ravel()] * 2
              + [(0, 0) if f else (0, None) for f in fa.ravel()] * 2)

    v = np.zeros(2 * n_w + 2 * n_a)
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            sol = sopt.minimize(func, v, args=(rho, alpha), method="L-BFGS-B",
                                jac=True, bounds=bounds)
            W_new, _ = unpack(sol.x)
            h_new, _ = h_func(W_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        v, h = sol.x, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W, A = unpack(v)
    W[np.abs(W) < w_threshold] = 0
    A[np.abs(A) < w_threshold] = 0
    return W, A
