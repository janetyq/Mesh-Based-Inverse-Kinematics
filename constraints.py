import numpy as np

# Constraints are a tuple (constrained_vertices, constrained_locations):
#   constrained_vertices:  (k,) vertex indices
#   constrained_locations: (k, 3) where those vertices must end up

def constrain_vertices_in_place(vertices, start=0, end=None):
    if end is None:
        end = len(vertices)
    constrained_vertices = np.arange(start, end)
    constrained_locations = vertices[start:end]
    return constrained_vertices, constrained_locations

def combine_constraints(constraints1, constraints2):
    constrained_vertices1, constrained_locations1 = constraints1
    constrained_vertices2, constrained_locations2 = constraints2
    constrained_vertices = np.concatenate((constrained_vertices1, constrained_vertices2))
    constrained_locations = np.concatenate((constrained_locations1, constrained_locations2))
    return constrained_vertices, constrained_locations

def constrained_indices(constraints, n):
    '''
    Indices into the unrolled vertex vector x (3n,) of the constrained coordinates
    (x coordinates first, then y, then z, matching get_x) and the values they must take.
    '''
    constrained_vertices, constrained_locations = constraints
    indices = np.concatenate([constrained_vertices + i*n for i in range(3)])
    values = constrained_locations.T.reshape(-1, 1)
    return indices, values

def constraint_preprocessing(constraints, n, G):
    '''
    Splits G x = f into G_tilda x_tilda + C = f where x_tilda holds only the free coordinates
    and C = G x_constrained is the contribution of the constrained (known) coordinates.
    '''
    indices, values = constrained_indices(constraints, n)
    G_tilda = np.delete(G, indices, axis=1)
    x_constrained = np.zeros((3*n, 1))
    x_constrained[indices] = values
    C = G @ x_constrained
    return G_tilda, C

def constraint_postprocessing(x_tilda, constraints, n):
    '''
    Rebuilds the full x (3n, 1) from the free coordinates x_tilda and the constrained values.
    '''
    indices, values = constrained_indices(constraints, n)
    free = np.ones(3*n, dtype=bool)
    free[indices] = False
    x = np.zeros((3*n, 1))
    x[free] = np.reshape(x_tilda, (-1, 1))
    x[indices] = values
    return x
