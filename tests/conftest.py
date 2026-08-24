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
def tube_meshes():
    from preprocessing import import_meshes

    return import_meshes(("tube.off", "curved_tube.off"))


@pytest.fixture(scope="session")
def tube_model(tube_meshes):
    """The tube example's precomputed model (no solve, no plotting)."""
    from model import build_model

    return build_model(*tube_meshes)


@pytest.fixture(scope="session")
def tube_constraints(tube_meshes):
    """The constraints used by meshik.py: front face held in place, last vertex moved to (8, 8, 0)."""
    from constraints import combine_constraints, constrain_vertices_in_place

    vertices = tube_meshes[0][0]
    front = constrain_vertices_in_place(vertices, start=0, end=8)
    end = (np.array([len(vertices) - 1]), np.array([[8.0, 8.0, 0.0]]))
    return combine_constraints(front, end)
