'''
Generates the figures in images/ used by the README.

Run from the repo root:  uv run python examples/make_readme_figures.py
'''
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constraints import combine_constraints, constrain_vertices_in_place  # noqa: E402
from matrices import compute_Mw_Dw  # noqa: E402
from methods import nonlinear_optimization, solve_for_constraints, solve_for_target_feature  # noqa: E402
from model import build_model  # noqa: E402
from preprocessing import import_meshes  # noqa: E402
from utils.plotting import plot_mesh_overlay, plot_mesh_row, x_to_vertices  # noqa: E402

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
EXAMPLE, GENERATED = 'Blues', 'Greens'


def save(fig, name):
    path = os.path.join(IMAGES, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('wrote', path)


def tube_constraints(vertices, end_location):
    '''front face held in place, last vertex moved to end_location'''
    front = constrain_vertices_in_place(vertices, start=0, end=8)
    end = (np.array([len(vertices) - 1]), np.array([end_location]))
    return combine_constraints(front, end)


def tube_figures():
    vertices_list, faces = import_meshes(('tube.off', 'curved_tube.off'))
    model = build_model(vertices_list, faces)
    examples = [x_to_vertices(x) for x in model.xs]

    # end vertex dragged from the straight example towards the curved example
    start, stop = vertices_list[0][-1], vertices_list[1][-1]
    generated = []
    for t in (0.25, 0.5, 0.75):
        constraints = tube_constraints(vertices_list[0], (1 - t) * start + t * stop)
        x, w, error = nonlinear_optimization(model, constraints, verbose=False)
        generated.append(x_to_vertices(x))
        print(f'  t={t}: w={w.ravel().round(3)} error={error:.3f}')

    fig = plot_mesh_row([examples[0], *generated, examples[1]], model.faces,
                        [EXAMPLE, GENERATED, GENERATED, GENERATED, EXAMPLE])
    save(fig, 'tube_ik_sweep.png')

    fig = plot_mesh_overlay([*examples, *generated], model.faces,
                            [EXAMPLE, EXAMPLE, 'Purples', 'Oranges', 'Reds'],
                            alphas=[0.15, 0.15, 0.6, 0.6, 0.6])
    save(fig, 'tube_ik_overlay.png')

    # same constraints, linear feature blend (paper eq. 4) vs nonlinear feature blend (paper eq. 8)
    constraints = tube_constraints(vertices_list[0], [8, 8, 0])
    x_linear, _, _ = solve_for_constraints(model, constraints)
    x_nonlinear, _, _ = nonlinear_optimization(model, constraints, verbose=False)
    fig = plot_mesh_row([examples[0], examples[1], x_to_vertices(x_linear), x_to_vertices(x_nonlinear)], model.faces,
                        [EXAMPLE, EXAMPLE, 'Oranges', GENERATED],
                        titles=['Example 1', 'Example 2', 'Linear feature blend', 'MeshIK'])
    save(fig, 'tube_linear_vs_nonlinear.png')

    # all five tube examples
    vertices_list, faces = import_meshes(('tube.off', 'bent_tube.off', 'curved_tube.off',
                                          'out_curved_tube.off', 'special_tube.off'))
    model = build_model(vertices_list, faces)
    fig = plot_mesh_row([x_to_vertices(x) for x in model.xs], model.faces, [EXAMPLE] * model.N,
                        options={'view': False}, shared_scale=False)
    save(fig, 'tube_examples.png')


def cat_figures():
    vertices_list, faces = import_meshes(('cat_000001.obj', 'cat_000011.obj'), mesh_dir=os.path.join('meshes', 'cat'))
    model = build_model(vertices_list, faces)
    examples = [x_to_vertices(x) for x in model.xs]

    # interpolation in feature space: target feature M(w) with w = (1-t, t), one vertex pinned to fix translation
    pin = np.argmin(np.linalg.norm(vertices_list[0] - vertices_list[0].mean(axis=0), axis=1))
    generated = []
    for t in (0.25, 0.5, 0.75):
        w = np.array([[1 - t], [t]])
        f_target, _ = compute_Mw_Dw(w, model.log_rotations, model.shears)
        location = (1 - t) * vertices_list[0][pin] + t * vertices_list[1][pin]
        x = solve_for_target_feature(model, (np.array([pin]), location[np.newaxis]), f_target)
        generated.append(x_to_vertices(x))
        print(f'  t={t}: interpolated')

    fig = plot_mesh_row([examples[0], *generated, examples[1]], model.faces,
                        [EXAMPLE, GENERATED, GENERATED, GENERATED, EXAMPLE])
    save(fig, 'cat_interpolation.png')


if __name__ == '__main__':
    os.makedirs(IMAGES, exist_ok=True)
    print('tube figures')
    tube_figures()
    print('cat figures')
    cat_figures()
