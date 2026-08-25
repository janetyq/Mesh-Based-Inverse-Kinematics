# ALL THE METHODS OF SOLVING
#
# Every solver takes a MeshIKModel and a constraints tuple (constrained_vertices, constrained_locations)
# and returns the full unrolled vertex vector x of shape (3n, 1), see matrices.get_x.
import numpy as np
import scipy.sparse as sp

from constraints import constraint_postprocessing, constraint_preprocessing
from matrices import compute_Mw_Dw


def solve_for_target_feature(model, constraints, f_target):
    '''
    Finds best vertex positions to match a given feature vector (9m, 1)
    '''
    G_tilda, C = constraint_preprocessing(constraints, model.n, model.G)
    # normal equations; G_tilda^T G_tilda is sparse and positive definite once translation is constrained
    x = sp.linalg.spsolve(G_tilda.T @ G_tilda, G_tilda.T @ (f_target - C)).reshape(-1, 1)
    return constraint_postprocessing(x, constraints, model.n)


def solve_for_constraints(model, constraints, k=0):
    '''
    Finds best linear blend of example meshes that satisfies the constrained vertices (paper eq. 4-5).
    Features are modelled as mean_f + M w, so w measures deviation from the mean example;
    k > 0 penalizes large w (pulls the solution towards the mean).
    '''
    G_tilda, C = constraint_preprocessing(constraints, model.n, model.G)
    n_tilda = G_tilda.shape[1] // 3
    M = model.feature_vectors
    mean_f = np.mean(M, axis=1, keepdims=True)

    A = np.hstack([G_tilda.toarray(), -M])  # dense: the linear solver is a small-mesh reference
    b = -C + mean_f
    if k != 0:
        Gamma = np.hstack([np.zeros((model.N, 3*n_tilda)), k * np.eye(model.N)])
        xw = np.linalg.solve(A.T @ A + Gamma.T @ Gamma, A.T @ b)
        error = np.linalg.norm(A @ xw - b)**2 + np.linalg.norm(Gamma @ xw)**2
    else:
        xw = np.linalg.pinv(A) @ b
        error = np.linalg.norm(A @ xw - b)**2
    x_result = constraint_postprocessing(xw[:3*n_tilda], constraints, model.n)
    w_result = xw[3*n_tilda:]
    return x_result, w_result, error


def nonlinear_optimization(model, constraints, max_iterations=10, tolerance=1e-4, verbose=True):
    '''
    Finds the best nonlinear blend of example meshes that satisfies the constrained vertices (paper eq. 8-9)
    by Gauss-Newton iteration, solving each linearized least squares problem with lsqr.
    '''
    G_tilda, C = constraint_preprocessing(constraints, model.n, model.G)
    n_tilda = G_tilda.shape[1] // 3
    w_result = np.ones((model.N, 1)) / model.N

    last_error = float("inf")
    iterations = 0
    while iterations < max_iterations:
        Mw, Dw = compute_Mw_Dw(w_result, model.log_rotations, model.shears)
        A = sp.hstack([G_tilda, -sp.csr_matrix(Dw)], format="csr")
        b = -C + Mw
        xw, istop, itn, error = sp.linalg.lsqr(A, b)[:4]
        w_result += xw[3*n_tilda:, np.newaxis]
        iterations += 1

        error_change = error - last_error
        if abs(error_change) < tolerance:
            break
        last_error = error

        if verbose:
            print("error", error)

    x_result = constraint_postprocessing(xw[:3*n_tilda], constraints, model.n)
    return x_result, w_result, error
