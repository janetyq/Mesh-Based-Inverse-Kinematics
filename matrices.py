
import numpy as np
from scipy.linalg import polar, expm
from math import cos, sin, pi, acos, asin, sqrt
import scipy.sparse as sp

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
    R /= np.linalg.det(R)
    if np.allclose(R, np.eye(3)):
        return np.zeros((3, 3))
    tr = max(min(np.trace(R), 3), -1)
    theta = acos((tr - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros((3, 3))
    elif np.isclose(theta, pi):
        return np.zeros((3, 3))
    K = 1 / (2 * sin(theta)) * (R - R.T)
    return theta * K

def compute_Mw_Dw(w, M_rotations, M_shears):    
    N, m = len(M_rotations), len(M_rotations[0])
    log_rotations = np.array([[calculate_rotation_log(R) for R in M_rotation] for M_rotation in M_rotations])
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