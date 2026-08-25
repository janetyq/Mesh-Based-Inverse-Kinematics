
import numpy as np
from scipy.linalg import polar, expm
from scipy.spatial.transform import Rotation
from math import pi

# MATRICES AND VECTORS

def get_v(vertices, face):
    '''
    Returns a matrix of shape (3, 3) where the columns are the
    location of the vertices of the face relative to the fourth vertex
    '''
    points = vertices[face]
    return np.array([points[0]-points[3], points[1]-points[3], points[2]-points[3]])

def get_G(ref_vertices, faces):
    '''
    Returns a matrix G of shape (9*m, 3*n) where m is the number of faces, n is the number of vertices
    Applying G onto a vector x of unrolled vertices returns the feature vector f of shape (9*m, 1)
    '''
    m = len(faces)
    n = len(ref_vertices)
    g = np.zeros((3*m, n))
    for idx, face in enumerate(faces):
        v_inv = np.linalg.inv(get_v(ref_vertices, face))
        g[3*idx:3*idx+3, face] = np.concatenate((v_inv, -np.sum(v_inv, axis=1, keepdims=True)), axis=1)

    G = np.kron(np.eye(3), g)
    return G

def get_x(mesh):
    '''
    Returns the unrolled vector of vertices of mesh
    Shape (3*n, 1)
    x = (x_1, ... x_n, y_1, ... y_n, z_1, ... z_n).T
    '''
    
    return np.reshape(mesh.T, (-1, 1)) # (3*n, 1)

def get_transformation(ref_mesh, def_mesh, face):
    ref_v = get_v(ref_mesh, face).T
    def_v = get_v(def_mesh, face).T
    return def_v @ np.linalg.inv(ref_v)

def get_feature_vector(ref_mesh, def_mesh, faces):
    '''
    Calculates affine transformation matrix (feature) for each
    face and places it into feature vector
    
    Returns 9*m feature_vector (1d array)
    '''
    m = len(faces)
    feature_vector = np.zeros((3, 3*m))
    for i, face in enumerate(faces):
        face_feature = get_transformation(ref_mesh, def_mesh, face)
        feature_vector[:, 3*i:3*i+3] = face_feature
    feature_vector = feature_vector.reshape((1, 9*m))
    return feature_vector


# FEATURES <-> TRANSFORMATIONS

def convert_feature_to_transformations(feature):
    # return feature.reshape((3, 3*m)).T.reshape((m, 3, 3)).transpose((0, 2, 1))
    return feature.reshape((3, -1)).T.reshape((-1, 3, 3)).transpose((0, 2, 1))

def convert_transformations_to_feature(transformations):
    # return transformations.transpose((0, 2, 1)).reshape((3*m, 3)).T.reshape((9*m, 1))
    return transformations.transpose((0, 2, 1)).reshape((-1, 3)).T.reshape((-1, 1))


# CODE CHECKERS

def check_feature_vectors(vertices_list, faces, input_feature_vectors):
    '''
    Takes input_feature_vectors in shape (9*m, N) where each column is a feature vector
    and verifies that its correct.

    Returns True if correct, False otherwise.
    '''
    N = len(vertices_list)
    correct_feature_vectors =  [get_feature_vector(vertices_list[0], vertices, faces) for vertices in vertices_list]
    is_equal = all([np.allclose(input, correct) for (input, correct) in zip(input_feature_vectors.T, correct_feature_vectors)])
    print("f == Gx", is_equal)
    return is_equal

# POLAR DECOMPOSITION
def polar_decomposition(transformations):
    rotation_arrays, shear_arrays = zip(*(polar(T) for T in transformations))
    return np.array(rotation_arrays), np.array(shear_arrays)

def get_M_components(feature_vectors):
    N = len(feature_vectors[0])
    M_transformations = [convert_feature_to_transformations(f) for f in feature_vectors.T]
    M_rotations, M_shears = zip(*[polar_decomposition(M_transformations[i]) for i in range(N)])
    return np.array(M_rotations), np.array(M_shears)

def calculate_rotation_log(R):
    '''
    Matrix logarithm of rotations R (..., 3, 3): the skew-symmetric matrices (..., 3, 3) whose
    axis-angle vectors have angle in [0, pi]. Exact at pi and well conditioned near it.
    '''
    R = np.asarray(R)
    rotvecs = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))
    return skew(rotvecs)

def skew(v):
    '''
    Skew-symmetric matrices (..., 3, 3) of vectors v (..., 3), so that skew(v) @ u = v x u
    '''
    v = np.asarray(v)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return K

def unskew(K):
    return np.stack([K[..., 2, 1], K[..., 0, 2], K[..., 1, 0]], axis=-1)

def face_neighbours(faces):
    '''
    For each face, the indices of the other faces sharing at least one of its original three vertices
    '''
    vertex_faces = {}
    for i, face in enumerate(faces):
        for v in face[:3]:
            vertex_faces.setdefault(v, []).append(i)
    return [sorted({j for v in face[:3] for j in vertex_faces[v]} - {i}) for i, face in enumerate(faces)]

def get_log_rotations(M_rotations, faces=None):
    '''
    Log of every rotation, shape (N, m, 3, 3). Depends only on the examples, so computed once.

    A rotation by exactly pi has two equally valid logs (axis n or -n). Blending a face whose log
    uses the opposite sign from its neighbours tears the mesh, so for those faces the sign is chosen
    to agree with the neighbouring faces (paper footnote 1). faces (m, 4) enables this; without it
    the sign is chosen to agree with the mesh as a whole.
    '''
    log_rotations = calculate_rotation_log(np.asarray(M_rotations))
    rotvecs = unskew(log_rotations)
    angles = np.linalg.norm(rotvecs, axis=-1)
    neighbours = face_neighbours(faces) if faces is not None else None
    for i in range(len(rotvecs)):
        at_pi = np.isclose(angles[i], pi, atol=1e-6)
        if not at_pi.any():
            continue
        others = rotvecs[i][~at_pi & (angles[i] > 0.5)]
        reference = others.mean(axis=0) if len(others) else np.zeros(3)
        for j in np.where(at_pi)[0]:
            ref = reference
            if neighbours is not None:
                near = [k for k in neighbours[j] if not at_pi[k] and angles[i][k] > 0.5]
                if near:
                    ref = rotvecs[i][near].mean(axis=0)
            if rotvecs[i][j] @ ref < 0:
                rotvecs[i][j] *= -1
    return skew(rotvecs)

def compute_Mw_Dw(w, log_rotations, M_shears):
    '''
    Nonlinear blend of the example features with weights w (N, 1) (paper eq. 6) and its
    Jacobian with respect to w (paper eq. 7).

    Returns Mw of shape (9m, 1) and Dw of shape (9m, N).
    '''
    N, m = len(log_rotations), len(log_rotations[0])
    weighted_log_rotations = np.sum(log_rotations  * w[:, np.newaxis, np.newaxis], axis=0)
    
    rotation_combo = np.array([np.real(expm(log_rot)) for log_rot in weighted_log_rotations])
    shear_combo = np.sum(M_shears * w[:, np.newaxis, np.newaxis], axis=0)

    Mw = convert_transformations_to_feature(np.array([R @ S for (R, S) in zip(rotation_combo, shear_combo)]))
    Dw = np.zeros((m, N, 3, 3))
    for k in range(N):
        for j in range(m):
            Dw[j, k] = rotation_combo[j] @ np.real(log_rotations[k, j]) @ shear_combo[j] + rotation_combo[j] @ M_shears[k, j]
    Dw = np.array([convert_transformations_to_feature(transformations) for transformations in Dw.transpose((1, 0, 2, 3))])
    Dw = Dw.transpose((1, 0, 2)).reshape((9*m, N))
    return Mw, Dw