import numpy as np
from utils.io import *

# MESH PREPROCESSING
def import_meshes(mesh_files):
    vertices_list = []
    faces = None
    for file in mesh_files:
        vertices, faces_ = read_obj(file) if file[-3:] == 'obj' else read_off(file)
        if faces is None:
            faces = faces_
        else:
            # check that all meshes have the same faces
            assert(np.array_equal(faces, faces_))
        vertices_list.append(vertices)       
        
    return np.array(vertices_list), faces

def calc_fourth_vertex(vertices, face):
    v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    cross_prod = np.cross(v2 - v1, v3 - v1)
    return v1 + cross_prod / np.linalg.norm(cross_prod)

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

    new_vertices_list = []
    for vertices in vertices_list:
        fourth_vertices = []
        for face in faces:
            fourth_vertices.append(calc_fourth_vertex(vertices, face))
        new_vertices_list.append(np.concatenate((vertices, fourth_vertices)))
    new_vertices_list = np.array(new_vertices_list)
    new_faces = np.concatenate((faces, np.reshape(np.arange(nv, n), (nf, 1))), axis=1)  
    return new_vertices_list, new_faces
