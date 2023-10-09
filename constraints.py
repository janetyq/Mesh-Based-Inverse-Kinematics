import numpy as np

def constrain_vertices_in_place(vertices, start=0, end=None):
    if end is None:
        end = len(vertices)
    constrained_vertices = np.arange(start, end)
    constrained_locations = vertices[start:end]
    return constrained_vertices, constrained_locations

def combine_constraints(constraints1, constraints2):
    # assumes in order
    constrained_vertices1, constrained_locations1 = constraints1
    constrained_vertices2, constrained_locations2 = constraints2
    constrained_vertices = np.concatenate((constrained_vertices1, constrained_vertices2))
    constrained_locations = np.concatenate((constrained_locations1, constrained_locations2))
    return constrained_vertices, constrained_locations

def constraint_preprocessing(constraints, n, G):
    constrained_vertices, constrained_locations = constraints
    delete_indices = np.concatenate([constrained_vertices + i*n for i in range(3)])
    G_tilda = np.delete(G, delete_indices[:, np.newaxis], axis=1)
    x_constrained = np.zeros((3*n, 1))
    x_constrained[delete_indices] = constrained_locations.T.reshape(len(constrained_vertices)*3, 1)
    C = G @ x_constrained
    return G_tilda, C

def constraint_postprocessing(x, constraints, n):
    constrained_vertices, constrained_locations = constraints
    for j in range(3):
        for i, v_idx in enumerate(constrained_vertices):
            x = np.insert(x, v_idx + j*n, constrained_locations[i, j], axis=0)
    return x