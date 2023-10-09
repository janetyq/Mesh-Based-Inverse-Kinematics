from utils.plotting import *
import numpy as np
from preprocessing import *
from matrices import *
from constraints import *
import scipy.sparse as sp

def nonlinear_optimization(constraints, verbose=True):
    constrained_vertices, constrained_locations = constraints
    G_tilda, C = constraint_preprocessing(constraints, n, G)
    G_tilda = sp.csr_matrix(G_tilda)
    n_tilda = n - len(constrained_vertices)
    w_result = np.ones((N, 1)) / N

    last_error = float("inf")
    iterations = 0
    while iterations < 10:
        Mw, Dw = compute_Mw_Dw(w_result, M_rotations, M_shears)
        A = sp.hstack([G_tilda, -sp.csr_matrix(Dw)], format="csr")
        b = -C + Mw
        xw, istop, itn, error = sp.linalg.lsqr(A, b)[:4]
        w_result += xw[3*n_tilda:, np.newaxis]
        iterations += 1

        error_change = error - last_error
        if abs(error_change) < 10**-4:
            break
        last_error = error
        
        if verbose:
            print("error", error)

    x_result = constraint_postprocessing(xw[:3*n_tilda], constraints, n)
    return x_result, w_result, error


# INPUT FROM OTHER FILES
MESH_FILES = "tube.off", "curved_tube.off"
vertices_list, faces = import_meshes(MESH_FILES)

# constraints
origin_constraints = (np.array([0]), np.array([[0, 0, 0]]))
front_face_constraints = constrain_vertices_in_place(vertices_list[0], start=0, end=8)
end_vertex, end_location = np.array([len(vertices_list[0])-1]), np.array([[8, 8, 0]])
end_constraint = (end_vertex, end_location)
constraints = combine_constraints(front_face_constraints, end_constraint)
constrained_vertices, constrained_locations = constraints

# add fourth vertex
vertices_list, faces = add_fourth_vertices(vertices_list, faces)

# get sizes
N, m, n = len(vertices_list), len(faces), len(vertices_list[0])

# calculate feature extractor G
reference_vertices = vertices_list[0]
G = get_G(reference_vertices, faces)
xs = np.array([get_x(vertices) for vertices in vertices_list])

# calculate feature vectors f (9m x N) (N feature vectors of height 9m)
feature_vectors = np.array([G@x for x in xs]).reshape(N, -1).T
# assert(check_feature_vectors(vertices_list, faces, feature_vectors)) # verify feature vectors are correct 

M_rotations, M_shears = get_M_components(feature_vectors)
x_result, w_result, error = nonlinear_optimization(constraints, verbose=True)

# Plotting
plotting_options = {
    'axlim': True,
    'cmap': 'Blues',
}

for i in range(N):
    plot_x_mesh(xs[i], faces, options=plotting_options, title="Original {}".format(i+1))
plot_x_mesh(x_result, faces, options=plotting_options, title="Mesh IK Solution", scatter_indices=constrained_vertices)
plt.show()

print('done')