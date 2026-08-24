"""G is the feature extractor: G @ x must equal the per-face deformation gradients computed directly."""
import numpy as np

from matrices import (
    check_feature_vectors,
    convert_feature_to_transformations,
    convert_transformations_to_feature,
    get_G,
)


def test_G_matches_per_face_transformations(tube_model):
    assert check_feature_vectors(tube_model["vertices_list"], tube_model["faces"], tube_model["feature_vectors"])


def test_G_shape_and_sparsity(tube_model):
    G, faces, verts = tube_model["G"], tube_model["faces"], tube_model["vertices_list"][0]
    m, n = len(faces), len(verts)
    assert G.shape == (9 * m, 3 * n)
    nnz_per_row = (G != 0).sum(axis=1)
    # each row touches the 4 vertices of one face; fewer only when v_inv has exact zeros (axis-aligned tube)
    assert np.all((1 <= nnz_per_row) & (nnz_per_row <= 4))
    assert nnz_per_row.max() == 4


def test_reference_mesh_has_identity_features(tube_model):
    """The reference mesh's deformation gradients are all the identity."""
    T = convert_feature_to_transformations(tube_model["feature_vectors"][:, 0])
    np.testing.assert_allclose(T, np.broadcast_to(np.eye(3), T.shape), atol=1e-10)


def test_feature_transformation_roundtrip(tube_model):
    f = tube_model["feature_vectors"][:, 1:2]
    np.testing.assert_array_equal(convert_transformations_to_feature(convert_feature_to_transformations(f)), f)


def test_get_G_is_deterministic(tube_model):
    np.testing.assert_array_equal(get_G(tube_model["vertices_list"][0], tube_model["faces"]), tube_model["G"])
