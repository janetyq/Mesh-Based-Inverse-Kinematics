"""Dw returned by compute_Mw_Dw must be the derivative of Mw with respect to the weights (paper Eq. 7).

This guards the nonlinear feature-space code through vectorization and is a prerequisite for the
Gauss-Newton / Cholesky solver, which trusts Dw completely.
"""
import numpy as np
import pytest

from matrices import compute_Mw_Dw


@pytest.mark.parametrize("seed", [0, 1])
def test_Dw_matches_finite_differences(tube_model, seed):
    L, S, N = tube_model.log_rotations, tube_model.shears, tube_model.N
    rng = np.random.default_rng(seed)
    w = np.ones((N, 1)) / N + 0.2 * rng.standard_normal((N, 1))

    _, Dw = compute_Mw_Dw(w, L, S)
    h = 1e-6
    for k in range(N):
        e = np.zeros((N, 1))
        e[k] = h
        Mw_plus, _ = compute_Mw_Dw(w + e, L, S)
        Mw_minus, _ = compute_Mw_Dw(w - e, L, S)
        fd = ((Mw_plus - Mw_minus) / (2 * h)).ravel()
        np.testing.assert_allclose(Dw[:, k], fd, rtol=1e-4, atol=1e-5)  # central-difference noise ~1e-6


def test_compute_Mw_Dw_does_not_mutate_inputs(tube_model):
    L, S = tube_model.log_rotations.copy(), tube_model.shears.copy()
    compute_Mw_Dw(np.full((tube_model.N, 1), 0.5), tube_model.log_rotations, tube_model.shears)
    np.testing.assert_array_equal(tube_model.log_rotations, L)
    np.testing.assert_array_equal(tube_model.shears, S)


@pytest.mark.xfail(
    strict=True,
    reason="calculate_rotation_log (matrices.py) returns a zero log for 180-degree rotations, so the curved "
    "tube's 8 end-cap faces are treated as unrotated. Fixed by the Rodrigues log/exp in Phase 2.",
)
def test_Mw_at_one_hot_weights_reproduces_example(tube_model):
    """M(e_i) must return example i's feature vector exactly (exp(log R) S = R S = T)."""
    L, S, N, F = tube_model.log_rotations, tube_model.shears, tube_model.N, tube_model.feature_vectors
    for i in range(N):
        w = np.zeros((N, 1))
        w[i] = 1.0
        Mw, _ = compute_Mw_Dw(w, L, S)
        np.testing.assert_allclose(Mw.ravel(), F[:, i], atol=1e-8)
