"""Constraint elimination and re-insertion are inverses, and G̃ x̃ + C == G x for any x honouring the constraints."""
import numpy as np

from constraints import (
    combine_constraints,
    constrain_vertices_in_place,
    constraint_postprocessing,
    constraint_preprocessing,
)


def _tube_constraints(tube_model):
    verts = tube_model["vertices_list"][0]
    front = constrain_vertices_in_place(verts, start=0, end=8)
    end = (np.array([len(verts) - 1]), np.array([[8.0, 8.0, 0.0]]))
    return combine_constraints(front, end)


def test_pre_post_roundtrip(tube_model):
    G, n = tube_model["G"], len(tube_model["vertices_list"][0])
    constraints = _tube_constraints(tube_model)
    idx, loc = constraints
    G_tilda, C = constraint_preprocessing(constraints, n, G)
    assert G_tilda.shape == (G.shape[0], 3 * (n - len(idx)))

    # any x that honours the constraints: G x == G̃ x̃ + C
    rng = np.random.default_rng(0)
    verts = rng.standard_normal((n, 3))
    verts[idx] = loc
    x = verts.T.reshape(-1, 1)
    free = np.ones(3 * n, dtype=bool)
    free[np.concatenate([idx + i * n for i in range(3)])] = False
    x_tilda = x[free]
    np.testing.assert_allclose(G_tilda @ x_tilda + C, G @ x, atol=1e-12)
    np.testing.assert_array_equal(constraint_postprocessing(x_tilda, constraints, n), x)
