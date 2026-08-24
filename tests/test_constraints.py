"""Constraint elimination and re-insertion are inverses, and G~ x~ + C == G x for any x honouring the constraints."""
import numpy as np

from constraints import constrained_indices, constraint_postprocessing, constraint_preprocessing


def test_pre_post_roundtrip(tube_model, tube_constraints):
    G, n = tube_model.G, tube_model.n
    idx, loc = tube_constraints
    G_tilda, C = constraint_preprocessing(tube_constraints, n, G)
    assert G_tilda.shape == (G.shape[0], 3 * (n - len(idx)))

    # any x that honours the constraints: G x == G~ x~ + C
    rng = np.random.default_rng(0)
    verts = rng.standard_normal((n, 3))
    verts[idx] = loc
    x = verts.T.reshape(-1, 1)
    free = np.ones(3 * n, dtype=bool)
    free[constrained_indices(tube_constraints, n)[0]] = False
    x_tilda = x[free]
    np.testing.assert_allclose(G_tilda @ x_tilda + C, G @ x, atol=1e-12)
    np.testing.assert_array_equal(constraint_postprocessing(x_tilda, tube_constraints, n), x)


def test_postprocessing_is_order_independent(tube_model, tube_constraints):
    """Constraint order must not matter (the old np.insert loop assumed ascending vertex indices)."""
    n = tube_model.n
    idx, loc = tube_constraints
    perm = np.random.default_rng(1).permutation(len(idx))
    shuffled = (idx[perm], loc[perm])
    x_tilda = np.arange(3 * (n - len(idx)), dtype=float)
    np.testing.assert_array_equal(
        constraint_postprocessing(x_tilda, shuffled, n), constraint_postprocessing(x_tilda, tube_constraints, n)
    )
