import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # meshik.py calls plt.show(); keep it headless under pytest


def run_meshik():
    """Run meshik.py as a script and return its module globals (x_result, w_result, error, ...)."""
    import runpy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "FigureCanvasAgg is non-interactive"
        return runpy.run_path("meshik.py", run_name="__main__")


@pytest.fixture(scope="session")
def meshik_run():
    return run_meshik()


@pytest.fixture(scope="session")
def tube_model():
    """The tube example's mesh-derived quantities, built from the library functions (no solve, no plotting)."""
    from matrices import get_G, get_M_components, get_x
    from preprocessing import add_fourth_vertices, import_meshes

    vertices_list, faces = import_meshes(("tube.off", "curved_tube.off"))
    vertices_list, faces = add_fourth_vertices(vertices_list, faces)
    N = len(vertices_list)
    G = get_G(vertices_list[0], faces)
    xs = np.array([get_x(v) for v in vertices_list])
    feature_vectors = np.array([G @ x for x in xs]).reshape(N, -1).T
    M_rotations, M_shears = get_M_components(feature_vectors)
    return dict(
        vertices_list=vertices_list, faces=faces, N=N, G=G, xs=xs,
        feature_vectors=feature_vectors, M_rotations=M_rotations, M_shears=M_shears,
    )
