"""Tripwire: the tube example must keep producing exactly the output it did before the refactor."""
import os

import numpy as np

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "tube.npz")


def test_tube_example_matches_golden(meshik_run):
    golden = np.load(GOLDEN)
    np.testing.assert_allclose(meshik_run["x_result"], golden["x_result"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(meshik_run["w_result"], golden["w_result"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(meshik_run["error"], golden["error"], rtol=1e-7)


def test_constrained_vertices_are_honoured(meshik_run):
    """Constrained vertices must sit exactly at their prescribed locations in the solution."""
    n = meshik_run["n"]
    x = meshik_run["x_result"].reshape(3, n).T  # (n, 3)
    idx, loc = meshik_run["constraints"]
    np.testing.assert_allclose(x[idx], loc, atol=1e-12)
