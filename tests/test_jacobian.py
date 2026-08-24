"""Dw returned by compute_Mw_Dw must be the derivative of Mw with respect to the weights (paper Eq. 7).

This guards the nonlinear feature-space code through vectorization and is a prerequisite for the
Gauss-Newton / Cholesky solver, which trusts Dw completely.
"""
import numpy as np
import pytest

from matrices import compute_Mw_Dw


@pytest.mark.parametrize("seed", [0, 1])
def test_Dw_matches_finite_differences(tube_model, seed):
    R, S, N = tube_model["M_rotations"], tube_model["M_shears"], tube_model["N"]
    rng = np.random.default_rng(seed)
    w = np.ones((N, 1)) / N + 0.2 * rng.standard_normal((N, 1))

    _, Dw = compute_Mw_Dw(w, R, S)
    h = 1e-6
    for k in range(N):
        e = np.zeros((N, 1))
        e[k] = h
        Mw_plus, _ = compute_Mw_Dw(w + e, R, S)
        Mw_minus, _ = compute_Mw_Dw(w - e, R, S)
        fd = ((Mw_plus - Mw_minus) / (2 * h)).ravel()
        np.testing.assert_allclose(Dw[:, k], fd, rtol=1e-4, atol=1e-5)  # central-difference noise ~1e-6


@pytest.mark.xfail(
    strict=True,
    reason="calculate_rotation_log (matrices.py:107) returns a zero log for 180-degree rotations, so the curved "
    "tube's 8 end-cap faces are treated as unrotated. Fixed by the Rodrigues log/exp in Phase 2.",
)
def test_Mw_at_one_hot_weights_reproduces_example(tube_model):
    """M(e_i) must return example i's feature vector exactly (exp(log R) S = R S = T)."""
    R, S, N, F = tube_model["M_rotations"], tube_model["M_shears"], tube_model["N"], tube_model["feature_vectors"]
    for i in range(N):
        w = np.zeros((N, 1))
        w[i] = 1.0
        Mw, _ = compute_Mw_Dw(w, R, S)
        np.testing.assert_allclose(Mw.ravel(), F[:, i], atol=1e-8)
