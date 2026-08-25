import os

import numpy as np

from utils.io import read_obj, read_off

# MESH PREPROCESSING
def import_meshes(mesh_files, mesh_dir="meshes"):
    vertices_list = []
    faces = None
    for file in mesh_files:
        path = os.path.join(mesh_dir, file)
        vertices, faces_ = read_obj(path) if os.path.splitext(file)[1] == '.obj' else read_off(path)
        if faces is None:
            faces = faces_
        else:
            # check that all meshes have the same faces
            assert(np.array_equal(faces, faces_))
        vertices_list.append(vertices)

    return np.array(vertices_list), faces

def calc_fourth_vertex(vertices, faces):
    '''
    Fourth vertex of each face (m, 3): the first vertex moved one unit along the face normal
    '''
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross_prod = np.cross(v2 - v1, v3 - v1)
    return v1 + cross_prod / np.linalg.norm(cross_prod, axis=1, keepdims=True)

def add_fourth_vertices(vertices_list, faces):
    '''
    For each mesh, adds a fourth vertex to each face as described
    in Sumner and Popovic 2004.

    Returns meshes_vertices_list and faces with the new vertices added.
    '''
    N = len(vertices_list)     # number of meshes
    nv = len(vertices_list[0]) # number of vertices initially
    nf = len(faces)     # number of faces
    n = nv + nf         # number of vertices after adding 4th vertex to each face

    new_vertices_list = np.array([np.concatenate((vertices, calc_fourth_vertex(vertices, faces)))
                                  for vertices in vertices_list])
    new_faces = np.concatenate((faces, np.reshape(np.arange(nv, n), (nf, 1))), axis=1)
    return new_vertices_list, new_faces
