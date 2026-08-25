"""The rotation log: exact for general rotations, and branch-consistent at 180 degrees.

The curved tube's 8 end-cap faces are rotated by exactly pi. Their log is ambiguous (axis n or -n);
a sign that disagrees with the neighbouring faces tears the end cap when blending (the "pinched end").
"""
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from matrices import calculate_rotation_log, get_log_rotations, skew, unskew


def test_log_is_inverse_of_exp():
    rng = np.random.default_rng(0)
    rotvecs = rng.standard_normal((50, 3))
    rotvecs *= (rng.uniform(0, np.pi, 50) / np.linalg.norm(rotvecs, axis=1))[:, None]  # angles in [0, pi)
    R = Rotation.from_rotvec(rotvecs).as_matrix()
    np.testing.assert_allclose(unskew(calculate_rotation_log(R)), rotvecs, atol=1e-10)
    np.testing.assert_allclose(np.array([expm(K) for K in calculate_rotation_log(R)]), R, atol=1e-10)


def test_log_at_pi_and_identity():
    R_pi = np.diag([-1.0, -1.0, 1.0])  # 180 degrees about z
    K = calculate_rotation_log(R_pi)
    assert np.isclose(np.linalg.norm(unskew(K)), np.pi)
    np.testing.assert_allclose(expm(K), R_pi, atol=1e-12)
    np.testing.assert_array_equal(calculate_rotation_log(np.eye(3)), np.zeros((3, 3)))


def test_skew_unskew():
    v = np.array([1.0, 2.0, 3.0])
    u = np.array([-0.5, 0.25, 4.0])
    np.testing.assert_allclose(skew(v) @ u, np.cross(v, u))
    np.testing.assert_array_equal(unskew(skew(v)), v)


def test_end_cap_logs_agree_with_neighbours(tube_model):
    """Curved tube: all 8 end-cap faces are at pi, and their axes must point the same way as the side faces."""
    rotvecs = unskew(tube_model.log_rotations[1])
    angles = np.linalg.norm(rotvecs, axis=1)
    at_pi = np.isclose(angles, np.pi, atol=1e-6)
    assert at_pi.sum() == 8
    axis = rotvecs / np.maximum(angles, 1e-12)[:, None]
    side = axis[~at_pi & (angles > 0.5)].mean(axis=0)
    assert np.all(axis[at_pi] @ side > 0.99)


def test_without_faces_falls_back_to_global_reference(tube_model):
    from matrices import convert_feature_to_transformations, polar_decomposition

    rotations = np.array([polar_decomposition(convert_feature_to_transformations(f))[0]
                          for f in tube_model.feature_vectors.T])
    np.testing.assert_allclose(get_log_rotations(rotations), tube_model.log_rotations, atol=1e-12)
