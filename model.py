from dataclasses import dataclass

import numpy as np

from matrices import get_G, get_M_components, get_log_rotations, get_x
from preprocessing import add_fourth_vertices


@dataclass(frozen=True)
class MeshIKModel:
    '''
    Everything derived from the example meshes that the solvers need, computed once.

    Shapes use the paper's notation: N example meshes, m faces, n vertices
    (n includes the fourth vertex added to every face).
    '''
    faces: np.ndarray            # (m, 4) vertex indices, fourth column is the added vertex
    xs: np.ndarray               # (N, 3n, 1) example poses as unrolled vertex vectors, see get_x
    G: np.ndarray                # (9m, 3n) feature extractor, f = G x
    feature_vectors: np.ndarray  # (9m, N) one feature vector per example (column)
    log_rotations: np.ndarray    # (N, m, 3, 3) log of the rotation part of every face transformation
    shears: np.ndarray           # (N, m, 3, 3) shear/stretch part of every face transformation

    @property
    def N(self):
        return self.feature_vectors.shape[1]

    @property
    def m(self):
        return len(self.faces)

    @property
    def n(self):
        return self.G.shape[1] // 3


def build_model(vertices_list, faces):
    '''
    Builds a MeshIKModel from example meshes (N, n_verts, 3) sharing the same faces (m, 3).
    The first mesh is the reference pose.
    '''
    vertices_list, faces = add_fourth_vertices(vertices_list, faces)
    N = len(vertices_list)

    G = get_G(vertices_list[0], faces)
    xs = np.array([get_x(vertices) for vertices in vertices_list])
    feature_vectors = np.array([G @ x for x in xs]).reshape(N, -1).T

    M_rotations, M_shears = get_M_components(feature_vectors)
    log_rotations = get_log_rotations(M_rotations)
    return MeshIKModel(faces, xs, G, feature_vectors, log_rotations, M_shears)
