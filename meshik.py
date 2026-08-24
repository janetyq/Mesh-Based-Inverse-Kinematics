import matplotlib.pyplot as plt
import numpy as np

from constraints import combine_constraints, constrain_vertices_in_place
from methods import nonlinear_optimization
from model import build_model
from preprocessing import import_meshes
from utils.plotting import plot_x_mesh

if __name__ == "__main__":
    # INPUT FROM OTHER FILES
    MESH_FILES = "tube.off", "curved_tube.off"
    vertices_list, faces = import_meshes(MESH_FILES)

    # constraints: hold the front face in place, move the last vertex to (8, 8, 0)
    front_face_constraints = constrain_vertices_in_place(vertices_list[0], start=0, end=8)
    end_vertex, end_location = np.array([len(vertices_list[0])-1]), np.array([[8, 8, 0]])
    end_constraint = (end_vertex, end_location)
    constraints = combine_constraints(front_face_constraints, end_constraint)
    constrained_vertices, constrained_locations = constraints

    # precompute feature extractor G and the feature vectors of the examples
    model = build_model(vertices_list, faces)
    # assert(check_feature_vectors(vertices_list, faces, model.feature_vectors)) # verify feature vectors are correct

    x_result, w_result, error = nonlinear_optimization(model, constraints, verbose=True)

    # Plotting
    plotting_options = {
        'axlim': True,
        'cmap': 'Blues',
    }

    for i in range(model.N):
        plot_x_mesh(model.xs[i], model.faces, options=plotting_options, title="Original {}".format(i+1))
    plot_x_mesh(x_result, model.faces, options=plotting_options, title="Mesh IK Solution", scatter_indices=constrained_vertices)
    plt.show()

    print('done')
