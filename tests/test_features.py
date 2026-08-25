"""G is the feature extractor: G @ x must equal the per-face deformation gradients computed directly."""
import numpy as np
import scipy.sparse as sp

from matrices import (
    check_feature_vectors,
    convert_feature_to_transformations,
    convert_transformations_to_feature,
    get_G,
)
from preprocessing import add_fourth_vertices


def test_G_matches_per_face_transformations(tube_meshes, tube_model):
    vertices_list, faces = add_fourth_vertices(*tube_meshes)
    assert check_feature_vectors(vertices_list, faces, tube_model.feature_vectors)


def test_model_sizes(tube_model):
    assert (tube_model.N, tube_model.m, tube_model.n) == (2, 208, 106 + 208)
    assert tube_model.G.shape == (9 * tube_model.m, 3 * tube_model.n)
    assert tube_model.xs.shape == (2, 3 * tube_model.n, 1)
    assert tube_model.log_rotations.shape == tube_model.shears.shape == (2, 208, 3, 3)


def test_G_sparsity(tube_model):
    assert isinstance(tube_model.G, sp.csc_matrix)
    nnz_per_row = np.diff(tube_model.G.tocsr().indptr)
    # each row touches the 4 vertices of one face; fewer only when v_inv has exact zeros (axis-aligned tube)
    assert np.all((1 <= nnz_per_row) & (nnz_per_row <= 4))
    assert nnz_per_row.max() == 4


def test_reference_mesh_has_identity_features(tube_model):
    """The reference mesh's deformation gradients are all the identity."""
    T = convert_feature_to_transformations(tube_model.feature_vectors[:, 0])
    np.testing.assert_allclose(T, np.broadcast_to(np.eye(3), T.shape), atol=1e-10)


def test_feature_transformation_roundtrip(tube_model):
    f = tube_model.feature_vectors[:, 1:2]
    np.testing.assert_array_equal(convert_transformations_to_feature(convert_feature_to_transformations(f)), f)


def test_get_G_is_deterministic(tube_meshes, tube_model):
    vertices_list, faces = add_fourth_vertices(*tube_meshes)
    np.testing.assert_array_equal(get_G(vertices_list[0], faces).toarray(), tube_model.G.toarray())
