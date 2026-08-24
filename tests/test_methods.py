"""The solvers in methods.py, exercised through the library API rather than the meshik.py script."""
import numpy as np

from constraints import constrain_vertices_in_place
from methods import nonlinear_optimization, solve_for_constraints, solve_for_target_feature
from preprocessing import add_fourth_vertices


def _curved_tube_constraints(tube_meshes):
    """Front face and last vertex pinned where they are in the curved tube (example 1)."""
    vertices_list, _ = tube_meshes
    curved = vertices_list[1]
    front = constrain_vertices_in_place(curved, start=0, end=8)
    end = (np.array([len(curved) - 1]), curved[-1:])
    return np.concatenate([front[0], end[0]]), np.concatenate([front[1], end[1]])


def test_solve_for_target_feature_recovers_example(tube_meshes, tube_model):
    """Asking for example 1's feature vector with example 1's constraints must give back example 1."""
    constraints = _curved_tube_constraints(tube_meshes)
    x = solve_for_target_feature(tube_model, constraints, tube_model.feature_vectors[:, 1:2])
    np.testing.assert_allclose(x, tube_model.xs[1], atol=1e-8)


def test_solve_for_constraints_recovers_example(tube_meshes, tube_model):
    """With example 1's constraints the linear blend is exactly example 1: mean_f + M w with w = (-1/2, +1/2)."""
    constraints = _curved_tube_constraints(tube_meshes)
    x, w, error = solve_for_constraints(tube_model, constraints)
    np.testing.assert_allclose(x, tube_model.xs[1], atol=1e-8)
    np.testing.assert_allclose(w.ravel(), [-0.5, 0.5], atol=1e-8)
    assert error < 1e-12


def test_solve_for_constraints_regularized_pulls_towards_mean(tube_meshes, tube_model):
    constraints = _curved_tube_constraints(tube_meshes)
    _, w0, _ = solve_for_constraints(tube_model, constraints, k=0)
    _, w1, _ = solve_for_constraints(tube_model, constraints, k=10.0)
    assert np.linalg.norm(w1) < np.linalg.norm(w0)


def test_nonlinear_optimization_matches_script(tube_model, tube_constraints, meshik_run):
    x, w, error = nonlinear_optimization(tube_model, tube_constraints, verbose=False)
    np.testing.assert_allclose(x, meshik_run["x_result"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(w, meshik_run["w_result"], rtol=1e-7, atol=1e-9)
    assert error == meshik_run["error"]


def test_fourth_vertices_are_unit_normals(tube_meshes):
    vertices_list, faces = add_fourth_vertices(*tube_meshes)
    for verts in vertices_list:
        v = verts[faces]
        normal = v[:, 3] - v[:, 0]
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0)
        np.testing.assert_allclose(np.einsum("ij,ij->i", normal, v[:, 1] - v[:, 0]), 0.0, atol=1e-12)
